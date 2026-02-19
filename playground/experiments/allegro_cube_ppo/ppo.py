"""PPO algorithm implementation for Newton environments.

Supports both LSTM and MLP architectures matching the DextrEme paper (Section 4).

Paper architecture:
    Actor:  LSTM(1024, layer_norm) + MLP [512, 512] + ELU
    Critic: LSTM(2048, layer_norm) + MLP [1024, 512] + ELU
    Separate optimizers: actor lr=1e-4 (linear), critic lr=5e-5 (fixed)
"""

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


def _build_mlp(input_dim: int, hidden_dims: tuple, act_fn, use_layer_norm: bool) -> tuple[nn.Sequential, int]:
    """Build an MLP with optional layer normalization.

    Returns:
        (sequential_module, output_dim)
    """
    layers = []
    prev = input_dim
    for dim in hidden_dims:
        layers.append(layer_init(nn.Linear(prev, dim), std=1.0))
        if use_layer_norm:
            layers.append(nn.LayerNorm(dim))
        layers.append(act_fn())
        prev = dim
    return nn.Sequential(*layers), prev


class ActorCritic(nn.Module):
    """Separate actor-critic networks supporting LSTM and MLP architectures.

    LSTM mode (paper):
        Actor:  obs -> LSTM(1024) -> LayerNorm -> MLP[512,512] -> action_mean
        Critic: obs -> LSTM(2048) -> LayerNorm -> MLP[1024,512] -> value
    MLP mode (simplified):
        Actor:  obs -> MLP[512,512,256,128] with LayerNorm -> action_mean
        Critic: obs -> MLP[512,512,256,128] with LayerNorm -> value

    Both modes have completely separate actor/critic parameters (no shared network).
    """

    def __init__(self, num_obs: int, num_actions: int, config: PPOConfig):
        super().__init__()
        self.network_type = config.network_type
        self.num_obs = num_obs
        self.num_actions = num_actions
        self.use_layer_norm = config.use_layer_norm

        act_fn = nn.ELU

        if config.network_type == "lstm":
            self.actor_lstm_size = config.actor_lstm_size
            self.critic_lstm_size = config.critic_lstm_size

            # Actor: LSTM(1024) + LayerNorm + MLP [512, 512]
            self.actor_lstm = nn.LSTM(num_obs, config.actor_lstm_size, num_layers=1, batch_first=False)
            if config.use_layer_norm:
                self.actor_lstm_ln = nn.LayerNorm(config.actor_lstm_size)
            self.actor_mlp, actor_out = _build_mlp(
                config.actor_lstm_size, config.actor_mlp_dims, act_fn, config.use_layer_norm
            )
            self.actor_mean = layer_init(nn.Linear(actor_out, num_actions), std=0.01)

            # Critic: LSTM(2048) + LayerNorm + MLP [1024, 512]
            self.critic_lstm = nn.LSTM(num_obs, config.critic_lstm_size, num_layers=1, batch_first=False)
            if config.use_layer_norm:
                self.critic_lstm_ln = nn.LayerNorm(config.critic_lstm_size)
            self.critic_mlp, critic_out = _build_mlp(
                config.critic_lstm_size, config.critic_mlp_dims, act_fn, config.use_layer_norm
            )
            self.critic_value = layer_init(nn.Linear(critic_out, 1), std=1.0)

        else:  # MLP mode
            self.actor_lstm_size = 0
            self.critic_lstm_size = 0

            # Separate actor MLP
            self.actor_mlp, actor_out = _build_mlp(
                num_obs, config.mlp_hidden_dims, act_fn, config.use_layer_norm
            )
            self.actor_mean = layer_init(nn.Linear(actor_out, num_actions), std=0.01)

            # Separate critic MLP
            self.critic_mlp, critic_out = _build_mlp(
                num_obs, config.mlp_hidden_dims, act_fn, config.use_layer_norm
            )
            self.critic_value = layer_init(nn.Linear(critic_out, 1), std=1.0)

        # Learnable log_std (DextrEme: init 0.0 -> sigma=1.0 for wide exploration)
        self.actor_log_std = nn.Parameter(torch.full((num_actions,), config.init_sigma))

    @property
    def is_lstm(self) -> bool:
        return self.network_type == "lstm"

    def actor_parameters(self) -> list[nn.Parameter]:
        """Return actor parameters for separate optimizer."""
        params = list(self.actor_mlp.parameters()) + list(self.actor_mean.parameters()) + [self.actor_log_std]
        if self.is_lstm:
            params += list(self.actor_lstm.parameters())
            if self.use_layer_norm:
                params += list(self.actor_lstm_ln.parameters())
        return params

    def critic_parameters(self) -> list[nn.Parameter]:
        """Return critic parameters for separate optimizer."""
        params = list(self.critic_mlp.parameters()) + list(self.critic_value.parameters())
        if self.is_lstm:
            params += list(self.critic_lstm.parameters())
            if self.use_layer_norm:
                params += list(self.critic_lstm_ln.parameters())
        return params

    def init_hidden(self, num_envs: int, device: str) -> dict | None:
        """Initialize LSTM hidden states. Returns None for MLP."""
        if not self.is_lstm:
            return None
        return {
            "actor": (
                torch.zeros(1, num_envs, self.actor_lstm_size, device=device),
                torch.zeros(1, num_envs, self.actor_lstm_size, device=device),
            ),
            "critic": (
                torch.zeros(1, num_envs, self.critic_lstm_size, device=device),
                torch.zeros(1, num_envs, self.critic_lstm_size, device=device),
            ),
        }

    def reset_hidden(self, hidden: dict, env_ids: torch.Tensor) -> dict:
        """Zero hidden states for specified environments."""
        if hidden is None:
            return None
        hidden["actor"][0][:, env_ids] = 0
        hidden["actor"][1][:, env_ids] = 0
        hidden["critic"][0][:, env_ids] = 0
        hidden["critic"][1][:, env_ids] = 0
        return hidden

    def _sanitize_obs(self, obs: torch.Tensor) -> torch.Tensor:
        obs = torch.nan_to_num(obs, nan=0.0, posinf=10.0, neginf=-10.0)
        return torch.clamp(obs, -100.0, 100.0)

    def _get_action_std(self) -> torch.Tensor:
        log_std_clamped = torch.clamp(self.actor_log_std, min=-5.0, max=2.0)
        return torch.exp(log_std_clamped)

    def _actor_features(self, obs: torch.Tensor, hidden=None):
        """Compute actor features from observation.

        Args:
            obs: (B, obs_dim) for single step
            hidden: (h, c) LSTM state or None

        Returns:
            features: (B, mlp_out_dim)
            new_hidden: (h, c) or None
        """
        if self.is_lstm:
            obs_seq = obs.unsqueeze(0)  # (1, B, obs_dim)
            lstm_out, new_hidden = self.actor_lstm(obs_seq, hidden)
            if self.use_layer_norm:
                lstm_out = self.actor_lstm_ln(lstm_out)
            features = self.actor_mlp(lstm_out.squeeze(0))
            return features, new_hidden
        else:
            features = self.actor_mlp(obs)
            return features, None

    def _critic_features(self, obs: torch.Tensor, hidden=None):
        """Compute critic features from observation."""
        if self.is_lstm:
            obs_seq = obs.unsqueeze(0)
            lstm_out, new_hidden = self.critic_lstm(obs_seq, hidden)
            if self.use_layer_norm:
                lstm_out = self.critic_lstm_ln(lstm_out)
            features = self.critic_mlp(lstm_out.squeeze(0))
            return features, new_hidden
        else:
            features = self.critic_mlp(obs)
            return features, None

    def get_action_and_value(
        self, obs: torch.Tensor, action: torch.Tensor | None = None, hidden: dict | None = None
    ) -> tuple:
        """Single-step forward pass (for rollout collection and MLP training).

        Returns:
            action, log_prob, entropy, value, action_mean, action_std, new_hidden
        """
        obs = self._sanitize_obs(obs)
        actor_hidden = hidden["actor"] if hidden else None
        critic_hidden = hidden["critic"] if hidden else None

        # Actor
        actor_feat, new_actor_hidden = self._actor_features(obs, actor_hidden)
        action_mean = self.actor_mean(actor_feat)
        action_mean = torch.nan_to_num(action_mean, nan=0.0, posinf=1.0, neginf=-1.0)
        action_std = self._get_action_std()

        dist = Normal(action_mean, action_std)
        if action is None:
            action = dist.sample()
        log_prob = dist.log_prob(action).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)

        # Critic
        critic_feat, new_critic_hidden = self._critic_features(obs, critic_hidden)
        value = self.critic_value(critic_feat).squeeze(-1)

        new_hidden = None
        if hidden is not None:
            new_hidden = {"actor": new_actor_hidden, "critic": new_critic_hidden}

        return action, log_prob, entropy, value, action_mean, action_std, new_hidden

    def evaluate_actions_sequence(
        self,
        obs_seq: torch.Tensor,
        actions_seq: torch.Tensor,
        dones_seq: torch.Tensor,
        actor_hidden_init: tuple,
        critic_hidden_init: tuple,
    ) -> tuple:
        """Evaluate actions over a sequence for LSTM training (BPTT).

        Processes the sequence step by step, resetting hidden states at
        episode boundaries (where done=True).

        Args:
            obs_seq: (T, B, obs_dim)
            actions_seq: (T, B, act_dim)
            dones_seq: (T, B)
            actor_hidden_init: (h, c) each (1, B, actor_lstm_size)
            critic_hidden_init: (h, c) each (1, B, critic_lstm_size)

        Returns:
            log_probs (T*B,), entropy (T*B,), values (T*B,),
            action_means (T*B, act_dim), action_stds (act_dim,)
        """
        T, B = obs_seq.shape[:2]
        obs_seq = self._sanitize_obs(obs_seq)

        # --- Actor LSTM: step by step with done-based hidden reset ---
        actor_h, actor_c = actor_hidden_init
        actor_features_list = []
        for t in range(T):
            if t > 0:
                # Zero hidden for envs that terminated at previous step
                not_done = (1.0 - dones_seq[t - 1]).unsqueeze(0).unsqueeze(-1)
                actor_h = actor_h * not_done
                actor_c = actor_c * not_done
            lstm_out, (actor_h, actor_c) = self.actor_lstm(obs_seq[t : t + 1], (actor_h, actor_c))
            if self.use_layer_norm:
                lstm_out = self.actor_lstm_ln(lstm_out)
            actor_features_list.append(lstm_out.squeeze(0))

        actor_features = torch.stack(actor_features_list)  # (T, B, lstm_size)
        actor_mlp_out = self.actor_mlp(actor_features)  # (T, B, mlp_out)
        action_means = self.actor_mean(actor_mlp_out)  # (T, B, act_dim)
        action_means = torch.nan_to_num(action_means, nan=0.0, posinf=1.0, neginf=-1.0)

        action_stds = self._get_action_std()
        dist = Normal(action_means, action_stds)
        log_probs = dist.log_prob(actions_seq).sum(dim=-1)  # (T, B)
        entropy = dist.entropy().sum(dim=-1)  # (T, B)

        # --- Critic LSTM: step by step with done-based hidden reset ---
        critic_h, critic_c = critic_hidden_init
        critic_features_list = []
        for t in range(T):
            if t > 0:
                not_done = (1.0 - dones_seq[t - 1]).unsqueeze(0).unsqueeze(-1)
                critic_h = critic_h * not_done
                critic_c = critic_c * not_done
            lstm_out, (critic_h, critic_c) = self.critic_lstm(obs_seq[t : t + 1], (critic_h, critic_c))
            if self.use_layer_norm:
                lstm_out = self.critic_lstm_ln(lstm_out)
            critic_features_list.append(lstm_out.squeeze(0))

        critic_features = torch.stack(critic_features_list)  # (T, B, lstm_size)
        critic_mlp_out = self.critic_mlp(critic_features)
        values = self.critic_value(critic_mlp_out).squeeze(-1)  # (T, B)

        return (
            log_probs.reshape(-1),
            entropy.reshape(-1),
            values.reshape(-1),
            action_means.reshape(-1, action_means.shape[-1]),
            action_stds,
        )

    def get_value(self, obs: torch.Tensor, hidden: dict | None = None) -> tuple:
        """Get value estimate.

        Returns:
            (value, new_hidden)
        """
        obs = self._sanitize_obs(obs)
        critic_hidden = hidden["critic"] if hidden else None
        critic_feat, new_critic_hidden = self._critic_features(obs, critic_hidden)
        value = self.critic_value(critic_feat).squeeze(-1)
        new_hidden = None
        if hidden is not None:
            new_hidden = {"actor": hidden["actor"], "critic": new_critic_hidden}
        return value, new_hidden


class RolloutBuffer:
    """Buffer for storing rollout data. Supports both MLP and LSTM modes."""

    def __init__(
        self,
        num_steps: int,
        num_envs: int,
        num_obs: int,
        num_actions: int,
        device: str,
        actor_lstm_size: int = 0,
        critic_lstm_size: int = 0,
    ):
        self.num_steps = num_steps
        self.num_envs = num_envs
        self.device = device
        self.is_lstm = actor_lstm_size > 0

        # Standard storage
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

        # LSTM: store initial hidden state (at the start of rollout)
        if self.is_lstm:
            self.initial_actor_h = torch.zeros(1, num_envs, actor_lstm_size, device=device)
            self.initial_actor_c = torch.zeros(1, num_envs, actor_lstm_size, device=device)
            self.initial_critic_h = torch.zeros(1, num_envs, critic_lstm_size, device=device)
            self.initial_critic_c = torch.zeros(1, num_envs, critic_lstm_size, device=device)

        self.step = 0

    def set_initial_hidden(self, hidden: dict):
        """Store LSTM hidden states at the start of a rollout."""
        if hidden is None:
            return
        self.initial_actor_h.copy_(hidden["actor"][0])
        self.initial_actor_c.copy_(hidden["actor"][1])
        self.initial_critic_h.copy_(hidden["critic"][0])
        self.initial_critic_c.copy_(hidden["critic"][1])

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

    def get_batches(self, minibatch_size: int):
        """Generate random minibatches for MLP training."""
        batch_size = self.num_steps * self.num_envs

        obs_flat = self.observations.reshape(-1, self.observations.shape[-1])
        actions_flat = self.actions.reshape(-1, self.actions.shape[-1])
        action_means_flat = self.action_means.reshape(-1, self.action_means.shape[-1])
        log_probs_flat = self.log_probs.reshape(-1)
        advantages_flat = self.advantages.reshape(-1)
        returns_flat = self.returns.reshape(-1)
        values_flat = self.values.reshape(-1)

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

    def get_sequential_batches(self, minibatch_size: int):
        """Generate sequential minibatches for LSTM training.

        Preserves temporal ordering. Each batch is a subset of environments
        processed over the full sequence length (num_steps).

        minibatch_size is total timesteps per batch:
            envs_per_batch = minibatch_size // num_steps
        """
        envs_per_batch = minibatch_size // self.num_steps
        env_indices = torch.randperm(self.num_envs, device=self.device)

        for start in range(0, self.num_envs, envs_per_batch):
            end = min(start + envs_per_batch, self.num_envs)
            batch_env_ids = env_indices[start:end]

            yield (
                self.observations[:, batch_env_ids],  # (T, B, obs_dim)
                self.actions[:, batch_env_ids],  # (T, B, act_dim)
                self.log_probs[:, batch_env_ids],  # (T, B)
                self.advantages[:, batch_env_ids],  # (T, B)
                self.returns[:, batch_env_ids],  # (T, B)
                self.values[:, batch_env_ids],  # (T, B)
                self.action_means[:, batch_env_ids],  # (T, B, act_dim)
                self.dones[:, batch_env_ids],  # (T, B)
                (
                    self.initial_actor_h[:, batch_env_ids].contiguous(),
                    self.initial_actor_c[:, batch_env_ids].contiguous(),
                ),
                (
                    self.initial_critic_h[:, batch_env_ids].contiguous(),
                    self.initial_critic_c[:, batch_env_ids].contiguous(),
                ),
            )

    def reset(self):
        """Reset buffer for new rollout."""
        self.step = 0


class PPO:
    """Proximal Policy Optimization with separate actor/critic optimizers.

    Paper: actor lr=1e-4 with linear schedule, critic lr=5e-5 fixed.
    Supports both "linear" (paper best) and "adaptive" (rl_games KL-based) LR schedules.
    """

    def __init__(
        self,
        actor_critic: ActorCritic,
        config: PPOConfig,
        device: str = "cuda",
    ):
        self.actor_critic = actor_critic.to(device)
        self.config = config
        self.device = device

        # Separate optimizers (paper: different LR for actor and critic)
        self.actor_optimizer = optim.Adam(actor_critic.actor_parameters(), lr=config.learning_rate)
        self.critic_optimizer = optim.Adam(actor_critic.critic_parameters(), lr=config.critic_learning_rate)

        # Value normalization (DextrEme: normalize_value = True)
        self.value_normalizer = RunningMeanStd(device=device) if config.normalize_value else None

    def update(self, buffer: RolloutBuffer, update_num: int = 0, total_updates: int = 1) -> dict[str, float]:
        """Update policy using PPO algorithm.

        Args:
            buffer: Rollout buffer with collected data
            update_num: Current update number (for linear LR schedule)
            total_updates: Total number of updates (for linear LR schedule)

        Returns:
            Dictionary of training metrics
        """
        # Linear LR schedule for actor (applied at start of each update)
        if self.config.lr_schedule == "linear" and total_updates > 1:
            progress = update_num / total_updates
            new_actor_lr = max(self.config.learning_rate * (1.0 - progress), 1e-6)
            for pg in self.actor_optimizer.param_groups:
                pg["lr"] = new_actor_lr

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

        for epoch in range(self.config.num_epochs):
            epoch_kl = 0.0
            epoch_batches = 0

            if self.actor_critic.is_lstm:
                batch_iter = buffer.get_sequential_batches(self.config.minibatch_size)
            else:
                batch_iter = buffer.get_batches(self.config.minibatch_size)

            for batch_data in batch_iter:
                if self.actor_critic.is_lstm:
                    (
                        obs_seq,
                        actions_seq,
                        old_log_probs_seq,
                        advs_seq,
                        returns_seq,
                        old_values_seq,
                        old_mu_seq,
                        dones_seq,
                        actor_h0,
                        critic_h0,
                    ) = batch_data

                    new_log_probs, entropy, new_values, new_mu, new_sigma = (
                        self.actor_critic.evaluate_actions_sequence(
                            obs_seq, actions_seq, dones_seq, actor_h0, critic_h0
                        )
                    )

                    # Flatten old data for loss computation
                    old_log_probs = old_log_probs_seq.reshape(-1)
                    advs = advs_seq.reshape(-1)
                    returns = returns_seq.reshape(-1)
                    old_values = old_values_seq.reshape(-1)
                    old_mu = old_mu_seq.reshape(-1, old_mu_seq.shape[-1])
                else:
                    obs, actions, old_log_probs, advs, returns, old_values, old_mu = batch_data

                    _, new_log_probs, entropy, new_values, new_mu, new_sigma, _ = (
                        self.actor_critic.get_action_and_value(obs, actions)
                    )

                # --- Policy loss (clipped surrogate objective) ---
                log_ratio = new_log_probs - old_log_probs
                ratio = torch.exp(log_ratio)
                pg_loss1 = -advs * ratio
                pg_loss2 = -advs * torch.clamp(ratio, 1 - self.config.clip_epsilon, 1 + self.config.clip_epsilon)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # --- Clipped value loss ---
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

                # --- Entropy loss ---
                entropy_loss = entropy.mean()

                # --- Bounds loss on action mean (rl_games style) ---
                soft_bound = 1.1
                mu_loss_high = torch.clamp_min(new_mu - soft_bound, 0.0) ** 2
                mu_loss_low = torch.clamp_max(new_mu + soft_bound, 0.0) ** 2
                bounds_loss = (mu_loss_high + mu_loss_low).sum(dim=-1).mean()

                # --- Separate losses for actor and critic ---
                actor_loss = pg_loss - self.config.entropy_coef * entropy_loss + self.config.bounds_loss_coef * bounds_loss
                critic_loss = self.config.value_coef * value_loss

                # --- Gradient steps ---
                self.actor_optimizer.zero_grad()
                self.critic_optimizer.zero_grad()

                actor_loss.backward()
                critic_loss.backward()

                nn.utils.clip_grad_norm_(self.actor_critic.actor_parameters(), self.config.max_grad_norm)
                nn.utils.clip_grad_norm_(self.actor_critic.critic_parameters(), self.config.max_grad_norm)

                self.actor_optimizer.step()
                self.critic_optimizer.step()

                # --- Analytical Gaussian KL divergence ---
                with torch.no_grad():
                    kl = gaussian_kl(old_mu, old_sigma.expand_as(old_mu), new_mu, new_sigma.expand_as(new_mu))

                # Accumulate metrics
                total_pg_loss += pg_loss.item()
                total_value_loss += value_loss.item()
                total_entropy_loss += entropy_loss.item()
                total_bounds_loss += bounds_loss.item()
                total_loss += (actor_loss + critic_loss).item()
                total_kl += kl.item()
                num_updates += 1
                epoch_kl += kl.item()
                epoch_batches += 1

            # Adaptive LR per epoch (only for actor, only if schedule is adaptive)
            if self.config.lr_schedule == "adaptive":
                avg_epoch_kl = epoch_kl / max(epoch_batches, 1)
                current_lr = self.actor_optimizer.param_groups[0]["lr"]
                if avg_epoch_kl > self.config.kl_threshold * 2.0:
                    new_lr = max(current_lr / 1.5, 1e-6)
                    for pg in self.actor_optimizer.param_groups:
                        pg["lr"] = new_lr
                elif avg_epoch_kl < self.config.kl_threshold * 0.5:
                    new_lr = min(current_lr * 1.5, 1e-2)
                    for pg in self.actor_optimizer.param_groups:
                        pg["lr"] = new_lr

        return {
            "policy_loss": total_pg_loss / max(num_updates, 1),
            "value_loss": total_value_loss / max(num_updates, 1),
            "entropy": total_entropy_loss / max(num_updates, 1),
            "bounds_loss": total_bounds_loss / max(num_updates, 1),
            "total_loss": total_loss / max(num_updates, 1),
            "kl_divergence": total_kl / max(num_updates, 1),
            "actor_learning_rate": self.actor_optimizer.param_groups[0]["lr"],
            "critic_learning_rate": self.critic_optimizer.param_groups[0]["lr"],
        }

    def save(self, path: str):
        """Save model checkpoint."""
        torch.save(
            {
                "actor_critic": self.actor_critic.state_dict(),
                "actor_optimizer": self.actor_optimizer.state_dict(),
                "critic_optimizer": self.critic_optimizer.state_dict(),
            },
            path,
        )

    def load(self, path: str):
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        self.actor_critic.load_state_dict(checkpoint["actor_critic"])
        self.actor_optimizer.load_state_dict(checkpoint["actor_optimizer"])
        self.critic_optimizer.load_state_dict(checkpoint["critic_optimizer"])
