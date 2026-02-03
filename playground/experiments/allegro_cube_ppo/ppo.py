"""PPO algorithm implementation (CleanRL style) for Newton environments."""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal

from .config import PPOConfig


def layer_init(layer: nn.Linear, std: float = 0.01) -> nn.Linear:
    """Initialize layer with orthogonal initialization."""
    nn.init.orthogonal_(layer.weight, std)
    nn.init.constant_(layer.bias, 0.0)
    return layer


class ActorCritic(nn.Module):
    """Actor-Critic network for continuous action spaces."""

    def __init__(
        self,
        num_obs: int,
        num_actions: int,
        hidden_dims: tuple[int, ...] = (256, 256, 128),
        activation: str = "elu",
    ):
        super().__init__()

        self.num_obs = num_obs
        self.num_actions = num_actions

        # Activation function
        if activation == "elu":
            act_fn = nn.ELU
        elif activation == "relu":
            act_fn = nn.ReLU
        elif activation == "tanh":
            act_fn = nn.Tanh
        else:
            act_fn = nn.ELU

        # Shared feature extractor
        layers = []
        prev_dim = num_obs
        for dim in hidden_dims:
            layers.append(layer_init(nn.Linear(prev_dim, dim), std=1.0))
            layers.append(act_fn())
            prev_dim = dim
        self.shared = nn.Sequential(*layers)

        # Actor head (mean)
        self.actor_mean = layer_init(nn.Linear(prev_dim, num_actions), std=0.01)

        # Actor log std (learnable parameter)
        self.actor_log_std = nn.Parameter(torch.zeros(num_actions))

        # Critic head
        self.critic = layer_init(nn.Linear(prev_dim, 1), std=1.0)

    def forward(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass returning action mean and value."""
        features = self.shared(obs)
        action_mean = self.actor_mean(features)
        value = self.critic(features)
        return action_mean, value

    def get_action_and_value(
        self, obs: torch.Tensor, action: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Get action, log prob, entropy, and value.

        Args:
            obs: Observation tensor
            action: If provided, compute log prob for this action

        Returns:
            action, log_prob, entropy, value
        """
        # Sanitize input observations
        obs = torch.nan_to_num(obs, nan=0.0, posinf=10.0, neginf=-10.0)
        obs = torch.clamp(obs, -100.0, 100.0)

        features = self.shared(obs)
        action_mean = self.actor_mean(features)

        # Clamp log_std to prevent numerical issues
        log_std_clamped = torch.clamp(self.actor_log_std, min=-20.0, max=2.0)
        action_std = torch.exp(log_std_clamped)

        # Replace NaN in action_mean with zeros
        action_mean = torch.nan_to_num(action_mean, nan=0.0, posinf=1.0, neginf=-1.0)

        dist = Normal(action_mean, action_std)

        if action is None:
            action = dist.sample()

        log_prob = dist.log_prob(action).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)
        value = self.critic(features).squeeze(-1)

        return action, log_prob, entropy, value

    def get_value(self, obs: torch.Tensor) -> torch.Tensor:
        """Get value estimate for observations."""
        # Sanitize input observations
        obs = torch.nan_to_num(obs, nan=0.0, posinf=10.0, neginf=-10.0)
        obs = torch.clamp(obs, -100.0, 100.0)

        features = self.shared(obs)
        return self.critic(features).squeeze(-1)


class RolloutBuffer:
    """Buffer for storing rollout data."""

    def __init__(self, num_steps: int, num_envs: int, num_obs: int, num_actions: int, device: str):
        self.num_steps = num_steps
        self.num_envs = num_envs
        self.device = device

        # Storage
        self.observations = torch.zeros(num_steps, num_envs, num_obs, device=device)
        self.actions = torch.zeros(num_steps, num_envs, num_actions, device=device)
        self.log_probs = torch.zeros(num_steps, num_envs, device=device)
        self.rewards = torch.zeros(num_steps, num_envs, device=device)
        self.dones = torch.zeros(num_steps, num_envs, device=device)
        self.values = torch.zeros(num_steps, num_envs, device=device)

        # Computed after rollout
        self.advantages = torch.zeros(num_steps, num_envs, device=device)
        self.returns = torch.zeros(num_steps, num_envs, device=device)

        self.step = 0

    def add(
        self,
        obs: torch.Tensor,
        action: torch.Tensor,
        log_prob: torch.Tensor,
        reward: torch.Tensor,
        done: torch.Tensor,
        value: torch.Tensor,
    ):
        """Add a step of data to the buffer."""
        self.observations[self.step] = obs
        self.actions[self.step] = action
        self.log_probs[self.step] = log_prob
        self.rewards[self.step] = reward
        self.dones[self.step] = done.float()
        self.values[self.step] = value
        self.step += 1

    def compute_returns_and_advantages(self, last_value: torch.Tensor, gamma: float, gae_lambda: float):
        """Compute returns and GAE advantages."""
        last_gae = 0

        for t in reversed(range(self.num_steps)):
            if t == self.num_steps - 1:
                next_value = last_value
                next_non_terminal = 1.0 - self.dones[t]
            else:
                next_value = self.values[t + 1]
                next_non_terminal = 1.0 - self.dones[t]

            delta = self.rewards[t] + gamma * next_value * next_non_terminal - self.values[t]
            last_gae = delta + gamma * gae_lambda * next_non_terminal * last_gae
            self.advantages[t] = last_gae

        self.returns = self.advantages + self.values

    def get_batches(self, num_minibatches: int):
        """Generate minibatches for training."""
        batch_size = self.num_steps * self.num_envs
        minibatch_size = batch_size // num_minibatches

        # Flatten data
        obs_flat = self.observations.reshape(-1, self.observations.shape[-1])
        actions_flat = self.actions.reshape(-1, self.actions.shape[-1])
        log_probs_flat = self.log_probs.reshape(-1)
        advantages_flat = self.advantages.reshape(-1)
        returns_flat = self.returns.reshape(-1)

        # Random permutation
        indices = torch.randperm(batch_size, device=self.device)

        for start in range(0, batch_size, minibatch_size):
            end = start + minibatch_size
            batch_indices = indices[start:end]

            yield (
                obs_flat[batch_indices],
                actions_flat[batch_indices],
                log_probs_flat[batch_indices],
                advantages_flat[batch_indices],
                returns_flat[batch_indices],
            )

    def reset(self):
        """Reset buffer for new rollout."""
        self.step = 0


class PPO:
    """Proximal Policy Optimization algorithm."""

    def __init__(
        self,
        actor_critic: ActorCritic,
        config: PPOConfig,
        device: str = "cuda",
    ):
        self.actor_critic = actor_critic.to(device)
        self.config = config
        self.device = device

        self.optimizer = optim.Adam(actor_critic.parameters(), lr=config.learning_rate)

    def update(self, buffer: RolloutBuffer) -> dict[str, float]:
        """Update policy using PPO algorithm.

        Args:
            buffer: Rollout buffer with collected data

        Returns:
            Dictionary of training metrics
        """
        # Normalize advantages
        advantages = buffer.advantages.reshape(-1)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        buffer.advantages = advantages.reshape(buffer.num_steps, buffer.num_envs)

        # Training metrics
        total_pg_loss = 0.0
        total_value_loss = 0.0
        total_entropy_loss = 0.0
        total_loss = 0.0
        num_updates = 0

        for _ in range(self.config.num_epochs):
            for obs, actions, old_log_probs, advs, returns in buffer.get_batches(self.config.num_minibatches):
                # Get current policy outputs
                _, new_log_probs, entropy, new_values = self.actor_critic.get_action_and_value(obs, actions)

                # Policy loss (clipped surrogate objective)
                log_ratio = new_log_probs - old_log_probs
                ratio = torch.exp(log_ratio)

                # Approximate KL divergence for monitoring
                # approx_kl = ((ratio - 1) - log_ratio).mean()

                pg_loss1 = -advs * ratio
                pg_loss2 = -advs * torch.clamp(ratio, 1 - self.config.clip_epsilon, 1 + self.config.clip_epsilon)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Value loss
                value_loss = 0.5 * ((new_values - returns) ** 2).mean()

                # Entropy loss (for exploration)
                entropy_loss = entropy.mean()

                # Total loss
                loss = pg_loss + self.config.value_coef * value_loss - self.config.entropy_coef * entropy_loss

                # Gradient step
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.actor_critic.parameters(), self.config.max_grad_norm)
                self.optimizer.step()

                # Accumulate metrics
                total_pg_loss += pg_loss.item()
                total_value_loss += value_loss.item()
                total_entropy_loss += entropy_loss.item()
                total_loss += loss.item()
                num_updates += 1

        return {
            "policy_loss": total_pg_loss / num_updates,
            "value_loss": total_value_loss / num_updates,
            "entropy": total_entropy_loss / num_updates,
            "total_loss": total_loss / num_updates,
        }

    def save(self, path: str):
        """Save model checkpoint."""
        torch.save(
            {
                "actor_critic": self.actor_critic.state_dict(),
                "optimizer": self.optimizer.state_dict(),
            },
            path,
        )

    def load(self, path: str):
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        self.actor_critic.load_state_dict(checkpoint["actor_critic"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
