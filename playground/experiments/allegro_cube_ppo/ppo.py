"""PPO algorithm implementation (CleanRL style) for Newton environments."""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal

from .config import PPOConfig


def gaussian_kl(mu0: torch.Tensor, sigma0: torch.Tensor, mu1: torch.Tensor, sigma1: torch.Tensor) -> torch.Tensor:
    """Analytical KL divergence between two diagonal Gaussians (rl_games style).

    Computes KL(p0 || p1) where p0 = N(mu0, sigma0^2), p1 = N(mu1, sigma1^2).
    """
    c1 = torch.log(sigma1 / (sigma0 + 1e-5) + 1e-5)
    c2 = (sigma0**2 + (mu0 - mu1) ** 2) / (2.0 * (sigma1**2 + 1e-5))
    c3 = -0.5
    kl = c1 + c2 + c3
    return kl.sum(dim=-1).mean()


class RunningMeanStd:
    """Running mean and standard deviation tracker (rl_games style).

    Used to normalize value targets so the critic trains on a stable scale.
    """

    def __init__(self, device: str = "cuda", epsilon: float = 1e-5):
        self.mean = torch.zeros((), dtype=torch.float32, device=device)
        self.var = torch.ones((), dtype=torch.float32, device=device)
        self.count = epsilon
        self.epsilon = epsilon

    def update(self, x: torch.Tensor):
        """Update running statistics with a batch of data."""
        batch_mean = x.mean()
        batch_var = x.var()
        batch_count = x.numel()

        delta = batch_mean - self.mean
        total_count = self.count + batch_count

        self.mean = self.mean + delta * batch_count / total_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        self.var = (m_a + m_b + delta**2 * self.count * batch_count / total_count) / total_count
        self.count = total_count

    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize input using running statistics."""
        return (x - self.mean) / (torch.sqrt(self.var) + self.epsilon)

    def denormalize(self, x: torch.Tensor) -> torch.Tensor:
        """Denormalize input using running statistics."""
        return x * torch.sqrt(self.var) + self.mean


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
        init_sigma: float = 0.0,
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
        # Initialize from config (DextrEme: 0.0 → σ=1.0 for wide exploration)
        self.actor_log_std = nn.Parameter(torch.full((num_actions,), init_sigma))

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
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Get action, log prob, entropy, value, action mean, and action std.

        Args:
            obs: Observation tensor
            action: If provided, compute log prob for this action

        Returns:
            action, log_prob, entropy, value, action_mean, action_std
        """
        # Sanitize input observations
        obs = torch.nan_to_num(obs, nan=0.0, posinf=10.0, neginf=-10.0)
        obs = torch.clamp(obs, -100.0, 100.0)

        features = self.shared(obs)
        action_mean = self.actor_mean(features)

        # Clamp log_std (rl_games style: wider range for exploration)
        # min=-5.0 → σ=0.007, max=2.0 → σ=7.39
        log_std_clamped = torch.clamp(self.actor_log_std, min=-5.0, max=2.0)
        action_std = torch.exp(log_std_clamped)

        # Replace NaN in action_mean with zeros
        action_mean = torch.nan_to_num(action_mean, nan=0.0, posinf=1.0, neginf=-1.0)

        dist = Normal(action_mean, action_std)

        if action is None:
            action = dist.sample()

        log_prob = dist.log_prob(action).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)
        value = self.critic(features).squeeze(-1)

        return action, log_prob, entropy, value, action_mean, action_std

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
        self.action_means = torch.zeros(num_steps, num_envs, num_actions, device=device)
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
        action_mean: torch.Tensor,
    ):
        """Add a step of data to the buffer."""
        self.observations[self.step] = obs
        self.actions[self.step] = action
        self.action_means[self.step] = action_mean
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
        action_means_flat = self.action_means.reshape(-1, self.action_means.shape[-1])
        log_probs_flat = self.log_probs.reshape(-1)
        advantages_flat = self.advantages.reshape(-1)
        returns_flat = self.returns.reshape(-1)
        values_flat = self.values.reshape(-1)

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
                values_flat[batch_indices],
                action_means_flat[batch_indices],
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

        # Value normalization (DextrEme: normalize_value = True)
        self.value_normalizer = RunningMeanStd(device=device) if config.normalize_value else None

    def update(self, buffer: RolloutBuffer) -> dict[str, float]:
        """Update policy using PPO algorithm (rl_games style).

        Matches rl_games behavior:
        - Analytical Gaussian KL divergence
        - No KL-based early stopping (rl_games doesn't use it)
        - Clipped value loss
        - Bounds loss on action mean (mu) with soft bound 1.1
        - Adaptive LR: /1.5 decrease, *1.5 increase, applied per epoch

        Args:
            buffer: Rollout buffer with collected data

        Returns:
            Dictionary of training metrics
        """
        # Normalize advantages
        advantages = buffer.advantages.reshape(-1)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        buffer.advantages = advantages.reshape(buffer.num_steps, buffer.num_envs)

        # Snapshot old sigma before any optimizer step changes it
        with torch.no_grad():
            old_log_std = torch.clamp(self.actor_critic.actor_log_std, min=-5.0, max=2.0)
            old_sigma = torch.exp(old_log_std)

        # Training metrics
        total_pg_loss = 0.0
        total_value_loss = 0.0
        total_entropy_loss = 0.0
        total_bounds_loss = 0.0
        total_loss = 0.0
        total_kl = 0.0
        num_updates = 0

        kl_threshold = self.config.kl_threshold
        lr_schedule = self.config.lr_schedule
        bounds_loss_coef = self.config.bounds_loss_coef

        for epoch in range(self.config.num_epochs):
            epoch_kl = 0.0
            epoch_batches = 0

            for obs, actions, old_log_probs, advs, returns, old_values, old_mu in buffer.get_batches(
                self.config.num_minibatches
            ):
                # Get current policy outputs (including mu and sigma)
                _, new_log_probs, entropy, new_values, new_mu, new_sigma = (
                    self.actor_critic.get_action_and_value(obs, actions)
                )

                # Policy loss (clipped surrogate objective)
                log_ratio = new_log_probs - old_log_probs
                ratio = torch.exp(log_ratio)

                pg_loss1 = -advs * ratio
                pg_loss2 = -advs * torch.clamp(ratio, 1 - self.config.clip_epsilon, 1 + self.config.clip_epsilon)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Clipped value loss (rl_games style)
                # Normalize returns and old_values if value normalization is enabled
                # (critic predicts in normalized space, so compare in normalized space)
                if self.value_normalizer is not None:
                    norm_returns = self.value_normalizer.normalize(returns)
                    norm_old_values = self.value_normalizer.normalize(old_values)
                else:
                    norm_returns = returns
                    norm_old_values = old_values

                value_pred_clipped = norm_old_values + torch.clamp(
                    new_values - norm_old_values,
                    -self.config.clip_epsilon,
                    self.config.clip_epsilon,
                )
                value_loss_unclipped = (new_values - norm_returns) ** 2
                value_loss_clipped = (value_pred_clipped - norm_returns) ** 2
                value_loss = 0.5 * torch.max(value_loss_unclipped, value_loss_clipped).mean()

                # Entropy loss
                entropy_loss = entropy.mean()

                # Bounds loss on action mean (rl_games style: penalize mu, soft bound 1.1)
                soft_bound = 1.1
                mu_loss_high = torch.clamp_min(new_mu - soft_bound, 0.0) ** 2
                mu_loss_low = torch.clamp_max(new_mu + soft_bound, 0.0) ** 2
                bounds_loss = (mu_loss_high + mu_loss_low).sum(dim=-1).mean()

                # Total loss
                loss = (
                    pg_loss
                    + self.config.value_coef * value_loss
                    - self.config.entropy_coef * entropy_loss
                    + bounds_loss_coef * bounds_loss
                )

                # Gradient step
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.actor_critic.parameters(), self.config.max_grad_norm)
                self.optimizer.step()

                # Analytical Gaussian KL divergence (rl_games style, no grad)
                with torch.no_grad():
                    kl = gaussian_kl(old_mu, old_sigma.expand_as(old_mu), new_mu, new_sigma.expand_as(new_mu))

                # Accumulate metrics
                total_pg_loss += pg_loss.item()
                total_value_loss += value_loss.item()
                total_entropy_loss += entropy_loss.item()
                total_bounds_loss += bounds_loss.item()
                total_loss += loss.item()
                total_kl += kl.item()
                num_updates += 1
                epoch_kl += kl.item()
                epoch_batches += 1

            # Adaptive LR per epoch (rl_games style: applied after each mini-epoch)
            avg_epoch_kl = epoch_kl / max(epoch_batches, 1)
            if lr_schedule == "adaptive":
                current_lr = self.optimizer.param_groups[0]["lr"]
                if avg_epoch_kl > kl_threshold * 2.0:
                    new_lr = max(current_lr / 1.5, 1e-6)
                    for param_group in self.optimizer.param_groups:
                        param_group["lr"] = new_lr
                elif avg_epoch_kl < kl_threshold * 0.5:
                    new_lr = min(current_lr * 1.5, 1e-2)
                    for param_group in self.optimizer.param_groups:
                        param_group["lr"] = new_lr

        avg_kl = total_kl / max(num_updates, 1)

        return {
            "policy_loss": total_pg_loss / max(num_updates, 1),
            "value_loss": total_value_loss / max(num_updates, 1),
            "entropy": total_entropy_loss / max(num_updates, 1),
            "bounds_loss": total_bounds_loss / max(num_updates, 1),
            "total_loss": total_loss / max(num_updates, 1),
            "kl_divergence": avg_kl,
            "learning_rate": self.optimizer.param_groups[0]["lr"],
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
