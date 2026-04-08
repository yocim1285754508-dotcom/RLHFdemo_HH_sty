import torch
import torch.nn as nn
import torch.nn.functional as F

from config import Config


class PPOAgent:
    def __init__(self, policy_container):
        self.policy = policy_container.model
        self.ref_model = policy_container.ref_model
        hidden_size = self._resolve_hidden_size(self.policy.config)
        policy_dtype = next(self.policy.parameters()).dtype
        self.value_head = nn.Linear(hidden_size, 1).to(device=Config.DEVICE, dtype=policy_dtype)
        self.optimizer = torch.optim.AdamW(
            list(self.policy.parameters()) + list(self.value_head.parameters()),
            lr=Config.LEARNING_RATE,
        )

        self.clip_ratio = 0.2
        self.ppo_epochs = 4
        self.mini_batch_size = 1

    def _resolve_hidden_size(self, config):
        for attr_name in ("hidden_size", "n_embd", "d_model"):
            value = getattr(config, attr_name, None)
            if isinstance(value, int) and value > 0:
                return value
        raise AttributeError(
            "Could not determine transformer hidden size from config. "
            "Expected one of: hidden_size, n_embd, d_model."
        )

    def get_policy_outputs(self, model, input_seq, response_ids_len):
        """
        Get token log-probabilities and hidden states for the generated
        response tokens only.
        """
        outputs = model(input_seq, output_hidden_states=True)
        logits = outputs.logits[:, :-1, :]
        hidden_states = outputs.hidden_states[-1][:, :-1, :]

        start_idx = input_seq.shape[1] - 1 - response_ids_len
        gen_logits = logits[:, start_idx:, :]
        gen_hidden_states = hidden_states[:, start_idx:, :]

        return F.log_softmax(gen_logits, dim=-1), gen_hidden_states

    def get_values(self, hidden_states):
        """
        Predict scalar values for each generated token.
        """
        hidden_states = hidden_states.to(self.value_head.weight.dtype)
        return self.value_head(hidden_states).squeeze(-1)

    def build_shaped_rewards(self, kl_div, raw_reward):
        """
        Apply per-token KL shaping and place the environment reward on the
        final generated token.
        """
        rewards = -Config.KL_COEF * kl_div
        rewards[:, -1] += raw_reward
        return rewards

    def compute_gae(self, rewards, values):
        """
        Generalized Advantage Estimation over the generated token sequence.
        """
        advantages = torch.zeros_like(rewards)
        last_advantage = torch.zeros_like(rewards[:, -1])
        next_value = torch.zeros_like(rewards[:, -1])

        for step in range(rewards.shape[1] - 1, -1, -1):
            delta = rewards[:, step] + Config.RETURN_GAMMA * next_value - values[:, step]
            last_advantage = (
                delta
                + Config.RETURN_GAMMA * Config.GAE_LAMBDA * last_advantage
            )
            advantages[:, step] = last_advantage
            next_value = values[:, step]

        returns = advantages + values

        if advantages.numel() > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        return returns, advantages

    def update(self, inputs, response_ids, raw_reward):
        """
        Run one PPO update using GAE and a learned value baseline.
        """
        full_seq = torch.cat([inputs, response_ids.unsqueeze(0)], dim=1)
        response_len = response_ids.shape[0]
        if response_len == 0:
            return 0.0, 0.0

        with torch.no_grad():
            old_log_probs_all, old_hidden_states = self.get_policy_outputs(
                self.policy,
                full_seq,
                response_len,
            )
            actions = response_ids.view(1, -1, 1)
            old_log_probs = old_log_probs_all.gather(2, actions).squeeze(-1)
            old_values = self.get_values(old_hidden_states)

            ref_log_probs_all, _ = self.get_policy_outputs(
                self.ref_model,
                full_seq,
                response_len,
            )
            ref_log_probs = ref_log_probs_all.gather(2, actions).squeeze(-1)

        kl_div = old_log_probs - ref_log_probs
        rewards = self.build_shaped_rewards(kl_div, raw_reward)
        returns, advantages = self.compute_gae(rewards, old_values)

        total_loss = 0.0
        for _ in range(self.ppo_epochs):
            new_log_probs_all, new_hidden_states = self.get_policy_outputs(
                self.policy,
                full_seq,
                response_len,
            )
            new_log_probs = new_log_probs_all.gather(2, actions).squeeze(-1)
            new_values = self.get_values(new_hidden_states)

            ratio = torch.exp(new_log_probs - old_log_probs)
            surr1 = ratio * advantages
            surr2 = torch.clamp(
                ratio,
                1.0 - self.clip_ratio,
                1.0 + self.clip_ratio,
            ) * advantages

            policy_loss = -torch.min(surr1, surr2).mean()
            value_loss = F.mse_loss(new_values, returns)
            loss = policy_loss + Config.VALUE_COEF * value_loss

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()

        return total_loss / self.ppo_epochs, kl_div.mean().item()
