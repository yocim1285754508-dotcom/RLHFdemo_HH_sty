import torch
import torch.nn.functional as F

from config import Config


class PPOAgent:
    def __init__(self, policy_container):
        self.policy = policy_container.model
        self.ref_model = policy_container.ref_model
        self.optimizer = torch.optim.AdamW(
            self.policy.parameters(),
            lr=Config.LEARNING_RATE,
        )

        self.clip_ratio = 0.2
        self.ppo_epochs = 4
        self.mini_batch_size = 1

    def get_log_probs(self, model, input_seq, response_ids_len):
        """
        Get log-probabilities for the generated response tokens only.
        """
        outputs = model(input_seq)
        logits = outputs.logits[:, :-1, :]

        start_idx = input_seq.shape[1] - 1 - response_ids_len
        gen_logits = logits[:, start_idx:, :]

        return F.log_softmax(gen_logits, dim=-1)

    def build_reward_to_go(self, old_log_probs, raw_reward):
        """
        Convert a terminal sequence-level reward into reward-to-go for every
        generated token:

            G_t = sum_{k=t}^{T-1} gamma^(k-t) * r_k

        Here we only have a terminal reward at the last step, so each position
        receives the discounted future return from that final reward.
        """
        response_len = old_log_probs.shape[1]
        if response_len == 0:
            return torch.zeros_like(old_log_probs)

        rewards = torch.zeros_like(old_log_probs)
        rewards[:, -1] = raw_reward

        returns = torch.zeros_like(rewards)
        running_return = torch.zeros_like(rewards[:, -1])

        for step in range(response_len - 1, -1, -1):
            running_return = rewards[:, step] + Config.RETURN_GAMMA * running_return
            returns[:, step] = running_return

        return returns

    def update(self, inputs, response_ids, raw_reward):
        """
        Run one PPO update using reward-to-go instead of terminal-only reward.
        """
        full_seq = torch.cat([inputs, response_ids.unsqueeze(0)], dim=1)
        response_len = response_ids.shape[0]

        with torch.no_grad():
            old_log_probs_all = self.get_log_probs(self.policy, full_seq, response_len)
            actions = response_ids.view(1, -1, 1)
            old_log_probs = old_log_probs_all.gather(2, actions).squeeze(-1)

            ref_log_probs_all = self.get_log_probs(self.ref_model, full_seq, response_len)
            ref_log_probs = ref_log_probs_all.gather(2, actions).squeeze(-1)

        kl_div = old_log_probs - ref_log_probs
        returns = self.build_reward_to_go(old_log_probs, raw_reward)

        # Token-level PPO objective with reward-to-go and per-token KL shaping.
        advantages = returns - Config.KL_COEF * kl_div

        if advantages.numel() > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        total_loss = 0.0
        for _ in range(self.ppo_epochs):
            new_log_probs_all = self.get_log_probs(self.policy, full_seq, response_len)
            new_log_probs = new_log_probs_all.gather(2, actions).squeeze(-1)

            ratio = torch.exp(new_log_probs - old_log_probs)
            surr1 = ratio * advantages
            surr2 = torch.clamp(
                ratio,
                1.0 - self.clip_ratio,
                1.0 + self.clip_ratio,
            ) * advantages

            loss = -torch.min(surr1, surr2).mean()

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()

        return total_loss / self.ppo_epochs, kl_div.mean().item()
