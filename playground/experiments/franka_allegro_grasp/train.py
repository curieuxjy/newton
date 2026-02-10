"""PPO training script for Franka + Allegro cube grasping with FABRICS.

This script trains a policy to grasp and lift a cube using:
1. PPO algorithm
2. FABRICS-based grasp features for enhanced observations/rewards
3. Multi-phase task structure (reach, grasp, lift)

Run with:
    uv run --extra examples --extra torch-cu12 python -m playground.experiments.franka_allegro_grasp.train

Reference:
- FABRICS: https://github.com/NVlabs/FABRICS
- CleanRL PPO: https://github.com/vwxyzjn/cleanrl
"""

import argparse
import os
import random
import time
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal
from torch.utils.tensorboard import SummaryWriter

from .config import EnvConfig, PPOConfig, TrainConfig
from .env import FrankaAllegroGraspEnv
from .fabric import GraspFabric


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

    Used to normalize value targets so the critic trains on a stable scale,
    preventing large swings in value_loss.
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
    """Actor-Critic network with optional FABRICS feature encoder."""

    def __init__(
        self,
        num_state_obs: int,
        num_fabric_obs: int,
        num_actions: int,
        hidden_dims: tuple[int, ...] = (512, 256, 128),
        activation: str = "elu",
    ):
        super().__init__()

        self.num_state_obs = num_state_obs
        self.num_fabric_obs = num_fabric_obs
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

        # FABRICS feature encoder (if using fabric features)
        self.fabric_encoder = None
        fabric_latent_dim = 0
        if num_fabric_obs > 0:
            fabric_latent_dim = 32
            self.fabric_encoder = nn.Sequential(
                layer_init(nn.Linear(num_fabric_obs, 64), std=1.0),
                act_fn(),
                layer_init(nn.Linear(64, fabric_latent_dim), std=1.0),
                act_fn(),
            )

        # Shared feature extractor
        total_input_dim = num_state_obs + fabric_latent_dim
        layers = []
        prev_dim = total_input_dim
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

    def _encode_features(
        self, state_obs: torch.Tensor, fabric_obs: torch.Tensor | None
    ) -> torch.Tensor:
        """Encode and concatenate observations."""
        if fabric_obs is not None and self.fabric_encoder is not None:
            fabric_features = self.fabric_encoder(fabric_obs)
            return torch.cat([state_obs, fabric_features], dim=-1)
        return state_obs

    def forward(
        self, state_obs: torch.Tensor, fabric_obs: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass returning action mean and value."""
        combined = self._encode_features(state_obs, fabric_obs)
        features = self.shared(combined)
        action_mean = self.actor_mean(features)
        value = self.critic(features)
        return action_mean, value

    def get_action_and_value(
        self,
        state_obs: torch.Tensor,
        fabric_obs: torch.Tensor | None = None,
        action: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Get action, log prob, entropy, value, action mean, and action std."""
        # Sanitize input
        state_obs = torch.nan_to_num(state_obs, nan=0.0, posinf=10.0, neginf=-10.0)
        state_obs = torch.clamp(state_obs, -100.0, 100.0)

        if fabric_obs is not None:
            fabric_obs = torch.nan_to_num(fabric_obs, nan=0.0, posinf=10.0, neginf=-10.0)
            fabric_obs = torch.clamp(fabric_obs, -100.0, 100.0)

        combined = self._encode_features(state_obs, fabric_obs)
        features = self.shared(combined)
        action_mean = self.actor_mean(features)

        # Clamp log_std (rl_games style)
        log_std_clamped = torch.clamp(self.actor_log_std, min=-5.0, max=2.0)
        action_std = torch.exp(log_std_clamped)

        action_mean = torch.nan_to_num(action_mean, nan=0.0, posinf=1.0, neginf=-1.0)

        dist = Normal(action_mean, action_std)

        if action is None:
            action = dist.sample()

        log_prob = dist.log_prob(action).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)
        value = self.critic(features).squeeze(-1)

        return action, log_prob, entropy, value, action_mean, action_std

    def get_value(
        self, state_obs: torch.Tensor, fabric_obs: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Get value estimate."""
        state_obs = torch.nan_to_num(state_obs, nan=0.0, posinf=10.0, neginf=-10.0)
        state_obs = torch.clamp(state_obs, -100.0, 100.0)

        if fabric_obs is not None:
            fabric_obs = torch.nan_to_num(fabric_obs, nan=0.0, posinf=10.0, neginf=-10.0)
            fabric_obs = torch.clamp(fabric_obs, -100.0, 100.0)

        combined = self._encode_features(state_obs, fabric_obs)
        features = self.shared(combined)
        return self.critic(features).squeeze(-1)


class RolloutBuffer:
    """Buffer for storing rollout data with FABRICS observations."""

    def __init__(
        self,
        num_steps: int,
        num_envs: int,
        num_state_obs: int,
        num_fabric_obs: int,
        num_actions: int,
        device: str,
    ):
        self.num_steps = num_steps
        self.num_envs = num_envs
        self.device = device

        # Storage
        self.state_observations = torch.zeros(num_steps, num_envs, num_state_obs, device=device)
        self.fabric_observations = (
            torch.zeros(num_steps, num_envs, num_fabric_obs, device=device)
            if num_fabric_obs > 0
            else None
        )
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
        state_obs: torch.Tensor,
        fabric_obs: torch.Tensor | None,
        action: torch.Tensor,
        log_prob: torch.Tensor,
        reward: torch.Tensor,
        done: torch.Tensor,
        value: torch.Tensor,
        action_mean: torch.Tensor,
    ):
        """Add a step of data."""
        self.state_observations[self.step] = state_obs
        if fabric_obs is not None and self.fabric_observations is not None:
            self.fabric_observations[self.step] = fabric_obs
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
        """Generate minibatches."""
        batch_size = self.num_steps * self.num_envs
        minibatch_size = batch_size // num_minibatches

        # Flatten data
        state_obs_flat = self.state_observations.reshape(-1, self.state_observations.shape[-1])
        fabric_obs_flat = (
            self.fabric_observations.reshape(-1, self.fabric_observations.shape[-1])
            if self.fabric_observations is not None
            else None
        )
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

            fabric_batch = fabric_obs_flat[batch_indices] if fabric_obs_flat is not None else None

            yield (
                state_obs_flat[batch_indices],
                fabric_batch,
                actions_flat[batch_indices],
                log_probs_flat[batch_indices],
                advantages_flat[batch_indices],
                returns_flat[batch_indices],
                values_flat[batch_indices],
                action_means_flat[batch_indices],
            )

    def reset(self):
        """Reset buffer."""
        self.step = 0


class PPOTrainer:
    """PPO trainer with FABRICS integration."""

    def __init__(
        self,
        env: FrankaAllegroGraspEnv,
        fabric: GraspFabric,
        config: TrainConfig,
        use_fabric_obs: bool = True,
    ):
        self.env = env
        self.fabric = fabric
        self.config = config
        self.use_fabric_obs = use_fabric_obs
        self.device = "cuda"

        # Fabric observation dimension
        # fingertip_dist(4) + closure(1) + palm_to_cube(3) + rewards(4) + fingertip_centroid(3) + ee_pos(3) = 18
        self.fabric_obs_dim = 18 if use_fabric_obs else 0

        # Create actor-critic
        self.actor_critic = ActorCritic(
            num_state_obs=env.num_state_obs,
            num_fabric_obs=self.fabric_obs_dim,
            num_actions=env.num_actions,
            hidden_dims=config.ppo.hidden_dims,
            activation=config.ppo.activation,
        ).to(self.device)

        # Optimizer
        self.optimizer = optim.Adam(
            self.actor_critic.parameters(),
            lr=config.ppo.learning_rate,
        )

        # Rollout buffer
        self.buffer = RolloutBuffer(
            num_steps=config.ppo.rollout_steps,
            num_envs=env.num_envs,
            num_state_obs=env.num_state_obs,
            num_fabric_obs=self.fabric_obs_dim,
            num_actions=env.num_actions,
            device=self.device,
        )

        # Value normalization (rl_games: normalize_value = True)
        self.value_normalizer = RunningMeanStd(device=self.device) if config.ppo.normalize_value else None

        # Logging
        self.writer = None

    def compute_fabric_obs(self) -> torch.Tensor | None:
        """Compute FABRICS-based observations."""
        if not self.use_fabric_obs:
            return None

        # Get current state
        body_q = torch.from_numpy(self.env.state_0.body_q.numpy()).to(self.device)
        joint_q = torch.from_numpy(self.env.state_0.joint_q.numpy()).to(self.device)

        body_q_reshaped = body_q.reshape(self.env.num_envs, self.env.bodies_per_env, 7)
        joint_q_reshaped = joint_q.reshape(self.env.num_envs, self.env.joint_q_per_env)

        # Extract relevant states
        ee_pos = body_q_reshaped[:, self.env.ee_body_idx, :3]
        ee_quat = body_q_reshaped[:, self.env.ee_body_idx, 3:7]
        cube_pos = body_q_reshaped[:, self.env.cube_body_idx, :3]
        allegro_q = joint_q_reshaped[
            :, self.env.allegro_dof_start : self.env.allegro_dof_start + self.env.allegro_dof_count
        ]

        # Compute grasp features
        grasp_features = self.fabric.compute_grasp_features(
            ee_pos, ee_quat, allegro_q, cube_pos, self.env.config.cube_size
        )

        # Compute grasp rewards
        grasp_rewards = self.fabric.compute_grasp_reward(
            grasp_features, self.env.config.cube_size
        )

        # Concatenate into observation
        fabric_obs = torch.cat(
            [
                grasp_features["fingertip_distances"],  # (batch, 4)
                grasp_features["grasp_closure"].unsqueeze(-1),  # (batch, 1)
                grasp_features["palm_to_cube"],  # (batch, 3)
                grasp_rewards["contact_reward"].unsqueeze(-1),  # (batch, 1)
                grasp_rewards["closure_reward"].unsqueeze(-1),  # (batch, 1)
                grasp_rewards["approach_reward"].unsqueeze(-1),  # (batch, 1)
                grasp_rewards["multi_contact_bonus"].unsqueeze(-1),  # (batch, 1)
                # Fingertip centroid (batch, 3)
                grasp_features["fingertip_positions"].mean(dim=1),
                # Palm position (batch, 3)
                ee_pos,
            ],
            dim=-1,
        )

        return fabric_obs

    def collect_rollout(self, state_obs: torch.Tensor) -> torch.Tensor:
        """Collect rollout data."""
        for _ in range(self.config.ppo.rollout_steps):
            # Compute fabric observations
            fabric_obs = self.compute_fabric_obs()

            # Get action
            with torch.no_grad():
                action, log_prob, _, value, action_mean, _ = self.actor_critic.get_action_and_value(
                    state_obs[:, : self.env.num_state_obs], fabric_obs
                )
                action = torch.clamp(action, -1.0, 1.0)
                # Denormalize value for GAE computation (critic predicts in normalized space)
                if self.value_normalizer is not None:
                    value = self.value_normalizer.denormalize(value)

            # Step environment
            next_obs, reward, done, info = self.env.step(action)

            # Accumulate reward components every env.step()
            for key, val in self.env.reward_components.items():
                if key not in self._rc_sums:
                    self._rc_sums[key] = 0.0
                self._rc_sums[key] += val
            self._rc_counts += 1

            # Track episode metrics
            self._episode_rewards += reward
            self._episode_lengths += 1

            # Handle episode completions
            done_indices = torch.where(done)[0]
            for idx in done_indices:
                self._completed_episodes += 1
                self._total_episode_reward += self._episode_rewards[idx].item()
                self._total_episode_length += self._episode_lengths[idx].item()
                self._episode_rewards[idx] = 0
                self._episode_lengths[idx] = 0

            # Store in buffer
            self.buffer.add(
                state_obs[:, : self.env.num_state_obs],
                fabric_obs,
                action,
                log_prob,
                reward,
                done,
                value,
                action_mean,
            )

            state_obs = next_obs

        return state_obs

    def update(self) -> dict[str, float]:
        """Update policy using PPO (rl_games style).

        Matches rl_games behavior:
        - Analytical Gaussian KL divergence
        - No KL-based early stopping (rl_games doesn't use it)
        - Clipped value loss
        - Bounds loss on action mean (mu) with soft bound 1.1
        - Adaptive LR: /1.5 decrease, *1.5 increase, applied per epoch
        """
        # Normalize advantages (DEXTRAH: normalize_advantage = True)
        if self.config.ppo.normalize_advantage:
            advantages = self.buffer.advantages.reshape(-1)
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
            self.buffer.advantages = advantages.reshape(
                self.buffer.num_steps, self.buffer.num_envs
            )

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

        ppo_cfg = self.config.ppo

        for epoch in range(ppo_cfg.num_epochs):
            epoch_kl = 0.0
            epoch_batches = 0

            for batch in self.buffer.get_batches(ppo_cfg.num_minibatches):
                state_obs, fabric_obs, actions, old_log_probs, advs, returns, old_values, old_mu = batch

                # Get current policy outputs (including mu and sigma)
                _, new_log_probs, entropy, new_values, new_mu, new_sigma = (
                    self.actor_critic.get_action_and_value(state_obs, fabric_obs, actions)
                )

                # Policy loss (clipped surrogate objective)
                log_ratio = new_log_probs - old_log_probs
                ratio = torch.exp(log_ratio)

                pg_loss1 = -advs * ratio
                pg_loss2 = -advs * torch.clamp(ratio, 1 - ppo_cfg.clip_epsilon, 1 + ppo_cfg.clip_epsilon)
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
                    new_values - norm_old_values, -ppo_cfg.clip_epsilon, ppo_cfg.clip_epsilon
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
                    + ppo_cfg.value_coef * value_loss
                    - ppo_cfg.entropy_coef * entropy_loss
                    + ppo_cfg.bounds_loss_coef * bounds_loss
                )

                # Gradient step
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.actor_critic.parameters(), ppo_cfg.max_grad_norm)
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
            if ppo_cfg.lr_schedule == "adaptive":
                current_lr = self.optimizer.param_groups[0]["lr"]
                if avg_epoch_kl > ppo_cfg.kl_threshold * 2.0:
                    new_lr = max(current_lr / 1.5, 1e-6)
                    for param_group in self.optimizer.param_groups:
                        param_group["lr"] = new_lr
                elif avg_epoch_kl < ppo_cfg.kl_threshold * 0.5:
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

    def train(self):
        """Main training loop."""
        # Setup logging
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_name = f"{self.config.experiment_name}_{timestamp}"
        log_dir = Path("runs") / run_name
        checkpoint_dir = Path(self.config.checkpoint_dir) / run_name

        log_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self.writer = SummaryWriter(log_dir)

        # Optional: Weights & Biases
        if self.config.wandb.enabled:
            try:
                import wandb

                wandb.init(
                    project=self.config.wandb.project,
                    entity=self.config.wandb.entity,
                    name=run_name,
                    config=asdict(self.config),
                    tags=self.config.wandb.tags,
                    mode=self.config.wandb.mode,
                )
            except ImportError:
                print("[WARNING] wandb not installed, skipping")
                self.config.wandb.enabled = False

        # Initialize environment
        state_obs = self.env.reset()
        global_step = 0
        num_updates = 0
        start_time = time.time()

        total_timesteps = self.config.ppo.total_timesteps
        batch_size = self.config.ppo.rollout_steps * self.env.num_envs

        best_reward = float("-inf")

        # Episode-based reward tracking (per-env accumulators)
        self._episode_rewards = torch.zeros(self.env.num_envs, device=self.device)
        self._episode_lengths = torch.zeros(self.env.num_envs, device=self.device, dtype=torch.int32)
        self._completed_episodes = 0
        self._total_episode_reward = 0.0
        self._total_episode_length = 0
        self._last_avg_reward = 0.0  # Carry forward when no episodes complete
        self._last_avg_length = 0.0

        # Reward component tracking (per env.step())
        self._rc_sums: dict[str, float] = {}
        self._rc_counts = 0

        print(f"[INFO] Starting training: {total_timesteps} timesteps")
        print(f"[INFO] Batch size: {batch_size}")
        print(f"[INFO] Checkpoints: {checkpoint_dir}")

        while global_step < total_timesteps:
            # Collect rollout (tracks episodes and reward components per env.step())
            state_obs = self.collect_rollout(state_obs)

            # Compute returns and advantages
            with torch.no_grad():
                fabric_obs = self.compute_fabric_obs()
                last_value = self.actor_critic.get_value(
                    state_obs[:, : self.env.num_state_obs], fabric_obs
                )
                # Denormalize last_value (critic predicts in normalized space)
                if self.value_normalizer is not None:
                    last_value = self.value_normalizer.denormalize(last_value)

            self.buffer.compute_returns_and_advantages(
                last_value, self.config.ppo.gamma, self.config.ppo.gae_lambda
            )

            # Update value normalizer with computed returns
            if self.value_normalizer is not None:
                self.value_normalizer.update(self.buffer.returns)

            # Update policy
            update_metrics = self.update()

            # Reset buffer
            self.buffer.reset()

            global_step += batch_size
            num_updates += 1

            # Logging
            if num_updates % self.config.log_interval == 0:
                elapsed = time.time() - start_time
                fps = global_step / elapsed

                # Episode metrics: use new data if available, otherwise carry forward
                if self._completed_episodes > 0:
                    avg_reward = self._total_episode_reward / self._completed_episodes
                    avg_length = self._total_episode_length / self._completed_episodes
                    self._last_avg_reward = avg_reward
                    self._last_avg_length = avg_length
                else:
                    avg_reward = self._last_avg_reward
                    avg_length = self._last_avg_length

                avg_reward_components = {}
                if self._rc_counts > 0:
                    for key, val in self._rc_sums.items():
                        avg_reward_components[key] = val / self._rc_counts

                print(
                    f"[{global_step:>10}] "
                    f"FPS: {fps:.0f} | "
                    f"reward: {avg_reward:.3f} | "
                    f"length: {avg_length:.1f} | "
                    f"episodes: {self._completed_episodes} | "
                    f"policy: {update_metrics['policy_loss']:.4f} | "
                    f"value: {update_metrics['value_loss']:.4f} | "
                    f"kl: {update_metrics['kl_divergence']:.4f} | "
                    f"lr: {update_metrics['learning_rate']:.2e}"
                )

                # TensorBoard
                self.writer.add_scalar("charts/fps", fps, global_step)
                self.writer.add_scalar("charts/learning_rate", update_metrics["learning_rate"], global_step)
                self.writer.add_scalar("episode/reward_mean", avg_reward, global_step)
                self.writer.add_scalar("episode/length_mean", avg_length, global_step)
                # self.writer.add_scalar("episode/completed", self._completed_episodes, global_step)
                self.writer.add_scalar("losses/policy_loss", update_metrics["policy_loss"], global_step)
                self.writer.add_scalar("losses/value_loss", update_metrics["value_loss"], global_step)
                self.writer.add_scalar("losses/entropy", update_metrics["entropy"], global_step)
                self.writer.add_scalar("losses/bounds_loss", update_metrics["bounds_loss"], global_step)
                self.writer.add_scalar("losses/kl_divergence", update_metrics["kl_divergence"], global_step)

                for key, value in avg_reward_components.items():
                    self.writer.add_scalar(f"reward/{key}", value, global_step)

                # Wandb
                if self.config.wandb.enabled:
                    wandb.log(
                        {
                            "charts/fps": fps,
                            "episode/reward_mean": avg_reward,
                            "episode/length_mean": avg_length,
                            # "episode/completed": self._completed_episodes,
                            **{f"losses/{k}": v for k, v in update_metrics.items()},
                            **{f"reward/{k}": v for k, v in avg_reward_components.items()},
                        },
                        step=global_step,
                    )

                # Reset episode and reward component tracking
                self._completed_episodes = 0
                self._total_episode_reward = 0.0
                self._total_episode_length = 0
                self._rc_sums = {}
                self._rc_counts = 0

            # Save best checkpoint
            if self._completed_episodes > 0:
                current_reward = self._total_episode_reward / self._completed_episodes
            else:
                current_reward = float("-inf")
            if current_reward > best_reward:
                best_reward = current_reward
                best_path = checkpoint_dir / "best.pt"
                self.save(best_path)
                print(f"[INFO] New best reward: {best_reward:.4f} -> saved {best_path}")

            # Save checkpoint
            if num_updates % self.config.save_interval == 0:
                checkpoint_path = checkpoint_dir / f"checkpoint_{global_step}.pt"
                self.save(checkpoint_path)
                print(f"[INFO] Saved checkpoint: {checkpoint_path}")

        # Save final model
        final_path = checkpoint_dir / "final.pt"
        self.save(final_path)
        print(f"[INFO] Training complete! Final model: {final_path}")

        self.writer.close()
        if self.config.wandb.enabled:
            wandb.finish()

    def save(self, path: str | Path):
        """Save checkpoint."""
        torch.save(
            {
                "actor_critic": self.actor_critic.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "config": asdict(self.config),
            },
            path,
        )

    def load(self, path: str | Path):
        """Load checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        self.actor_critic.load_state_dict(checkpoint["actor_critic"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])


def main():
    parser = argparse.ArgumentParser(description="Train Franka+Allegro Grasp with PPO and FABRICS")

    # Environment
    parser.add_argument("--num-envs", type=int, default=256, help="Number of environments")
    parser.add_argument("--use-depth", action="store_true", help="Enable depth sensor")
    parser.add_argument("--no-fabric", action="store_true", help="Disable FABRICS observations")

    # Training (DEXTRAH defaults)
    parser.add_argument("--total-timesteps", type=int, default=100_000_000, help="Total timesteps")
    parser.add_argument("--learning-rate", type=float, default=5e-4, help="Learning rate (DEXTRAH: 5e-4)")
    parser.add_argument("--rollout-steps", type=int, default=16, help="Rollout steps per update (DEXTRAH: horizon_length=16)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    # Logging
    parser.add_argument("--no-wandb", action="store_true", help="Disable wandb logging")
    parser.add_argument("--wandb-project", type=str, default="newton-franka-allegro-grasp")
    parser.add_argument("--log-interval", type=int, default=10, help="Log interval")
    parser.add_argument("--save-interval", type=int, default=100, help="Save interval")

    # Checkpoint
    parser.add_argument("--checkpoint", type=str, default=None, help="Resume from checkpoint")

    args = parser.parse_args()

    # Set random seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    # Create config
    env_config = EnvConfig(
        num_envs=args.num_envs,
        use_depth_sensor=args.use_depth,
        randomize_cube_pos=False,  # Disable for now to simplify
    )

    ppo_config = PPOConfig(
        total_timesteps=args.total_timesteps,
        learning_rate=args.learning_rate,
        rollout_steps=args.rollout_steps,
    )

    train_config = TrainConfig(
        seed=args.seed,
        log_interval=args.log_interval,
        save_interval=args.save_interval,
        env=env_config,
        ppo=ppo_config,
    )
    train_config.wandb.enabled = not args.no_wandb
    train_config.wandb.project = args.wandb_project

    print("=" * 60)
    print("Franka + Allegro Grasp Training with PPO and FABRICS")
    print("=" * 60)
    print(f"Environments: {args.num_envs}")
    print(f"Total timesteps: {args.total_timesteps}")
    print(f"Use FABRICS: {not args.no_fabric}")
    print(f"Use depth sensor: {args.use_depth}")
    print("=" * 60)

    # Create environment
    print("[INFO] Creating environment...")
    env = FrankaAllegroGraspEnv(env_config, device="cuda", headless=True)

    # Create FABRICS module
    print("[INFO] Creating FABRICS module...")
    fabric = GraspFabric(
        franka_dof=env.franka_dof_count,
        allegro_dof=env.allegro_dof_count,
        device="cuda",
    )

    # Create trainer
    print("[INFO] Creating trainer...")
    trainer = PPOTrainer(
        env=env,
        fabric=fabric,
        config=train_config,
        use_fabric_obs=not args.no_fabric,
    )

    # Resume from checkpoint if provided
    if args.checkpoint:
        print(f"[INFO] Loading checkpoint: {args.checkpoint}")
        trainer.load(args.checkpoint)

    # Train
    print("[INFO] Starting training...")
    trainer.train()


if __name__ == "__main__":
    main()
