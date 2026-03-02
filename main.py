# main.py
import torch
from config import Config
from models import LanguageModelPolicy
from discriminator import SafetyDiscriminator
from scheduler import FeedbackScheduler
from agent import PPOAgent
from utils import set_seed, get_human_input
from datasets import load_dataset

def get_hh_prompts(num_samples=10):
    """
    加载 Anthropic/hh-rlhf 数据集，并提取 Prompt
    """
    print("Loading Anthropic/hh-rlhf dataset...")
    # split="train[:100]" 表示只下载前100条，防止下载太久
    dataset = load_dataset("Anthropic/hh-rlhf", split=f"train[:{num_samples}]")
    
    prompts = []
    for item in dataset:
        # HH数据集的 'chosen' 字段包含完整的对话
        full_text = item['chosen']
        
        # 我们只需要提取 "Human: ... Assistant:" 之前的部分作为 Prompt
        # 简单逻辑：找到最后一个 "Assistant:"，截取到那里
        split_token = "Assistant:"
        if split_token in full_text:
            # rsplit 从右边找，确保处理多轮对话
            prompt = full_text.rsplit(split_token, 1)[0] + split_token
            prompts.append(prompt)
        else:
            # 容错处理
            prompts.append(full_text[:50]) 
            
    return prompts

def main():
    # 1. 初始化环境
    set_seed(Config.SEED)
    print("Initializing Hybrid Alignment System...")
    
    # 2. 实例化组件 (对应表格中的 Modules)
    policy = LanguageModelPolicy()            # Learner
    discriminator = SafetyDiscriminator()     # Internal Auditor
    scheduler = FeedbackScheduler()           # HR / Controller
    agent = PPOAgent(policy)                  # Optimizer
    
    prompts = get_hh_prompts(num_samples=10)
    
    print(f"\nStart Training Loop for {len(prompts)} episodes using HH Dataset...\n")
    print("-" * 50)
    
    print(f"\nStart Training Loop for {len(prompts)} episodes...\n")
    print("-" * 50)
    
    for epoch in range(Config.EPOCHS):
        for i, prompt_text in enumerate(prompts):
            # === Step 1: Rollout (学生写作业) ===
            response_ids, response_text, inputs_ids = policy.generate(prompt_text)
            full_text = prompt_text + response_text
            
            # === Step 2: Risk Assessment (判别器打分) ===
            risk_score, uncertainty = discriminator.predict(response_text)
            
            # === Step 3: Scheduling (调度器决定谁说了算) ===
            source_type, need_human = scheduler.decide(risk_score, uncertainty)
            
            # === Step 4: Obtain Feedback (获取信号) ===
            human_label = None
            if need_human:
                # 触发人类介入 (Human-in-the-loop)
                human_label = get_human_input(prompt_text, response_text)
                
                # === 修改点：使用 Replay Buffer 更新 ===
                # 每次只传当前这一条，Discriminator 内部会自动混合旧数据
                disc_loss = discriminator.update_with_replay(response_text, human_label, batch_size=8)
                
                print(f"  [Discriminator Update] Feedback received. Replay Loss: {disc_loss:.4f}")
            
            # === Step 5: Reward Aggregation (计算总分) ===
            raw_reward = scheduler.get_reward_source(source_type, risk_score, human_label)
            
            # === Step 6: Optimization (PPO 更新) ===
            loss, kl = agent.update(inputs_ids, response_ids, raw_reward)
            
            # === Logging ===
            print(f"Epoch {epoch} | Step {i}")
            print(f"  Prompt: {prompt_text}")
            print(f"  Response: {response_text}")
            print(f"  [Risk]: {risk_score:.2f} (Uncertainty: {uncertainty:.2f})")
            print(f"  [Control]: {source_type} -> Raw Reward: {raw_reward:.2f}")
            print(f"  [Update]: Loss: {loss:.4f} | KL: {kl:.4f}")
            print("-" * 30)

if __name__ == "__main__":
    main()

