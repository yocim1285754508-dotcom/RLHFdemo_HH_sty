import torch
import torch.nn.functional as F
from config import Config

class PPOAgent:
    def __init__(self, policy_container):
        self.policy = policy_container.model
        self.ref_model = policy_container.ref_model
        self.optimizer = torch.optim.AdamW(self.policy.parameters(), lr=Config.LEARNING_RATE)
        
        # === PPO 超参数 ===
        self.clip_ratio = 0.2    # ε: 允许策略变化的幅度
        self.ppo_epochs = 4      # 每次更新多少轮
        self.mini_batch_size = 1 

    def get_log_probs(self, model, input_seq, response_ids_len):
        """
        辅助函数：计算序列中生成的 token 的 log_prob
        [修复说明]: 修正了切片索引，确保长度与 response_ids 一致
        """
        outputs = model(input_seq)
        # logits 预测的是下一个词，所以整体左移
        # logits shape: [Batch, Seq_Len - 1, Vocab]
        logits = outputs.logits[:, :-1, :]
        
        # 我们只需要 response 部分的 logits
        # response 在 input_seq 的末尾
        # Start Index = 总长度 - response长度 - 1 (因为logits比input少1)
        # End Index = 末尾
        start_idx = input_seq.shape[1] - 1 - response_ids_len
        
        # === 修正点: 去掉了之前的 :-1，改为取到最后 ===
        gen_logits = logits[:, start_idx :, :]
        
        log_probs = F.log_softmax(gen_logits, dim=-1)
        return log_probs

    def update(self, inputs, response_ids, raw_reward):
        """
        执行 PPO 更新
        """
        # 1. 准备数据
        # 拼接 [Prompt + Response]
        # inputs: [1, seq_len]
        # response_ids: [resp_len] -> unsqueeze -> [1, resp_len]
        full_seq = torch.cat([inputs, response_ids.unsqueeze(0)], dim=1)
        response_len = response_ids.shape[0]
        
        # 2. 计算 "旧" 策略 (Old Policy)
        with torch.no_grad():
            old_log_probs_all = self.get_log_probs(self.policy, full_seq, response_len)
            
            # Gather: 提取实际生成 token 的概率
            # actions shape: [1, resp_len, 1]
            actions = response_ids.view(1, -1, 1)
            
            # 这里的 gather 要求 old_log_probs_all 的第1维长度必须等于 actions 的第1维长度
            old_log_probs = old_log_probs_all.gather(2, actions).squeeze(-1)
            
            # 参考模型
            ref_log_probs_all = self.get_log_probs(self.ref_model, full_seq, response_len)
            ref_log_probs = ref_log_probs_all.gather(2, actions).squeeze(-1)

        # 3. 计算优势 (Advantage)
        # KL Divergence
        kl_div = old_log_probs - ref_log_probs
        
        # 构建 Reward (Sparse Reward: 只给最后一个 token 打分)
        rewards = torch.zeros_like(old_log_probs)
        rewards[:, -1] = raw_reward 
        
        # Advantage = Reward - beta * KL
        advantages = rewards - Config.KL_COEF * kl_div
        
        # Normalize Advantage (训练稳定的关键)
        if advantages.numel() > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # 4. PPO 更新循环
        total_loss = 0
        for _ in range(self.ppo_epochs):
            # 新策略 (带梯度)
            new_log_probs_all = self.get_log_probs(self.policy, full_seq, response_len)
            new_log_probs = new_log_probs_all.gather(2, actions).squeeze(-1)
            
            # Ratio
            ratio = torch.exp(new_log_probs - old_log_probs)
            
            # Surrogate Loss
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1.0 - self.clip_ratio, 1.0 + self.clip_ratio) * advantages
            
            loss = -torch.min(surr1, surr2).mean()
            
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            
        return total_loss / self.ppo_epochs, kl_div.mean().item()