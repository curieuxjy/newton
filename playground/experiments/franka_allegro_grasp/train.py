"""Teacher PPO training with LSTM for Franka + Allegro cube grasping.

DEXTRAH-G Stage 1: Train a privileged teacher policy using full state
information (including object pose/velocity). No depth images.

Architecture:
  Actor:  teacher_obs -> LSTM(1024) -> MLP[512,512] -> action(11)
  Critic: teacher_obs -> LSTM(2048) -> MLP[1024,512] -> value(1)

Run with:
    uv run --extra examples --extra torch-cu12 python -m playground.experiments.franka_allegro_grasp.train

Reference:
- DEXTRAH-G: NVlabs/DEXTRAH
- rl_games configs: rl_games_ppo_lstm_cfg.yaml
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
from torch.utils.tensorboard import SummaryWriter

from .config import EnvConfig, TeacherPPOConfig, TrainConfig
from .env import FrankaAllegroGraspEnv
from .fabric import GraspFabric
from .networks import ObsRunningMeanStd, RunningMeanStd, TeacherActorCritic, gaussian_kl


class SequenceRolloutBuffer:
    """Rollout buffer for LSTM-based PPO with sequence support.

    Stores transitions and generates sequence-based minibatches
    for LSTM training.
    """

    def __init__(
        self,
        num_steps: int,
        num_envs: int,
        num_actor_obs: int,
        num_critic_obs: int,
        num_actions: int,
        device: str,
    ):
        self.num_steps = num_steps
        self.num_envs = num_envs
        self.device = device

        # Storage
        self.actor_obs = torch.zeros(num_steps, num_envs, num_actor_obs, device=device)
        self.critic_obs = torch.zeros(num_steps, num_envs, num_critic_obs, device=device)
        self.actions = torch.zeros(num_steps, num_envs, num_actions, device=device)
        self.action_means = torch.zeros(num_steps, num_envs, num_actions, device=device)
        self.log_probs = torch.zeros(num_steps, num_envs, device=device)
        self.rewards = torch.zeros(num_steps, num_envs, device=device)
        self.dones = torch.zeros(num_steps, num_envs, device=device)
        self.timeouts = torch.zeros(num_steps, num_envs, device=device)
        self.values = torch.zeros(num_steps, num_envs, device=device)

        # LSTM hidden states at the start of each step
        # Shape: (num_steps, num_layers, num_envs, hidden_dim)
        self.actor_h = None
        self.actor_c = None
        self.critic_h = None
        self.critic_c = None

        # Computed after rollout
        self.advantages = torch.zeros(num_steps, num_envs, device=device)
        self.returns = torch.zeros(num_steps, num_envs, device=device)

        self.step = 0

    def store_hidden(self, actor_hidden, critic_hidden, step: int):
        """Store LSTM hidden states for a given step."""
        ah, ac = actor_hidden
        ch, cc = critic_hidden

        if self.actor_h is None:
            # Initialize on first call
            self.actor_h = torch.zeros(
                self.num_steps, *ah.shape, device=self.device
            )
            self.actor_c = torch.zeros_like(self.actor_h)
            self.critic_h = torch.zeros(
                self.num_steps, *ch.shape, device=self.device
            )
            self.critic_c = torch.zeros_like(self.critic_h)

        self.actor_h[step] = ah.detach()
        self.actor_c[step] = ac.detach()
        self.critic_h[step] = ch.detach()
        self.critic_c[step] = cc.detach()

    def add(
        self,
        actor_obs: torch.Tensor,
        critic_obs: torch.Tensor,
        action: torch.Tensor,
        log_prob: torch.Tensor,
        reward: torch.Tensor,
        done: torch.Tensor,
        value: torch.Tensor,
        action_mean: torch.Tensor,
        timeout: torch.Tensor | None = None,
    ):
        self.actor_obs[self.step] = actor_obs
        self.critic_obs[self.step] = critic_obs
        self.actions[self.step] = action
        self.action_means[self.step] = action_mean
        self.log_probs[self.step] = log_prob
        self.rewards[self.step] = reward
        self.dones[self.step] = done.float()
        self.timeouts[self.step] = timeout.float() if timeout is not None else 0.0
        self.values[self.step] = value
        self.step += 1

    def compute_returns_and_advantages(self, last_value: torch.Tensor, gamma: float, gae_lambda: float):
        last_gae = 0
        for t in reversed(range(self.num_steps)):
            if t == self.num_steps - 1:
                next_value = last_value
            else:
                next_value = self.values[t + 1]
            # Only cut value chain on true termination (failure), not timeout.
            # For timeout: bootstrap from next value (continuation estimate).
            terminated = self.dones[t] - self.timeouts[t]  # 1 if failure, 0 if timeout or alive
            next_non_terminal = 1.0 - terminated
            delta = self.rewards[t] + gamma * next_value * next_non_terminal - self.values[t]
            # Reset GAE accumulation on ANY done (both terminated and timeout)
            # because the next step's observation is from a new episode.
            gae_non_terminal = 1.0 - self.dones[t]
            last_gae = delta + gamma * gae_lambda * gae_non_terminal * last_gae
            self.advantages[t] = last_gae
        self.returns = self.advantages + self.values

    def get_sequence_batches(self, seq_len: int, num_minibatches: int):
        """Generate sequence-based minibatches for LSTM training.

        Splits rollout into non-overlapping sequences of length seq_len.
        Shuffles environments (not time steps) to maintain temporal ordering.
        """
        assert self.num_steps % seq_len == 0 or self.num_steps == seq_len, (
            f"num_steps ({self.num_steps}) should be divisible by seq_len ({seq_len})"
        )

        num_sequences_per_env = max(1, self.num_steps // seq_len)
        total_sequences = num_sequences_per_env * self.num_envs
        sequences_per_batch = max(1, total_sequences // num_minibatches)

        # Shuffle environment indices
        env_indices = torch.randperm(self.num_envs, device=self.device)

        for batch_start in range(0, self.num_envs, self.num_envs // num_minibatches):
            batch_end = min(batch_start + self.num_envs // num_minibatches, self.num_envs)
            batch_envs = env_indices[batch_start:batch_end]

            for seq_start in range(0, self.num_steps, seq_len):
                seq_end = min(seq_start + seq_len, self.num_steps)
                actual_seq_len = seq_end - seq_start

                # Extract sequence data for selected environments
                # Shape: (actual_seq_len, batch_envs_count, ...)
                seq_actor_obs = self.actor_obs[seq_start:seq_end, batch_envs]
                seq_critic_obs = self.critic_obs[seq_start:seq_end, batch_envs]
                seq_actions = self.actions[seq_start:seq_end, batch_envs]
                seq_action_means = self.action_means[seq_start:seq_end, batch_envs]
                seq_log_probs = self.log_probs[seq_start:seq_end, batch_envs]
                seq_advantages = self.advantages[seq_start:seq_end, batch_envs]
                seq_returns = self.returns[seq_start:seq_end, batch_envs]
                seq_values = self.values[seq_start:seq_end, batch_envs]
                seq_dones = self.dones[seq_start:seq_end, batch_envs]

                # Get initial hidden states for this sequence
                # Shape: (num_layers, batch_envs_count, hidden_dim)
                if self.actor_h is not None:
                    seq_actor_h = self.actor_h[seq_start, :, batch_envs]
                    seq_actor_c = self.actor_c[seq_start, :, batch_envs]
                    seq_critic_h = self.critic_h[seq_start, :, batch_envs]
                    seq_critic_c = self.critic_c[seq_start, :, batch_envs]
                    actor_hidden = (seq_actor_h, seq_actor_c)
                    critic_hidden = (seq_critic_h, seq_critic_c)
                else:
                    actor_hidden = None
                    critic_hidden = None

                # Transpose to (batch, seq, ...) for LSTM
                yield {
                    "actor_obs": seq_actor_obs.transpose(0, 1),
                    "critic_obs": seq_critic_obs.transpose(0, 1),
                    "actions": seq_actions.transpose(0, 1),
                    "action_means": seq_action_means.transpose(0, 1),
                    "log_probs": seq_log_probs.transpose(0, 1),
                    "advantages": seq_advantages.transpose(0, 1),
                    "returns": seq_returns.transpose(0, 1),
                    "values": seq_values.transpose(0, 1),
                    "dones": seq_dones.transpose(0, 1),
                    "actor_hidden": actor_hidden,
                    "critic_hidden": critic_hidden,
                }

    def reset(self):
        self.step = 0


class TeacherTrainer:
    """PPO trainer for DEXTRAH-G teacher policy with LSTM."""

    def __init__(
        self,
        env: FrankaAllegroGraspEnv,
        config: TrainConfig,
    ):
        self.env = env
        self.config = config
        self.ppo = config.teacher
        self.device = config.device

        # Create actor-critic with asymmetric architecture
        self.actor_critic = TeacherActorCritic(
            num_actor_obs=env.num_teacher_obs,
            num_critic_obs=env.num_teacher_obs,  # Central value uses privileged obs
            num_actions=env.num_actions,
            actor_lstm_units=self.ppo.actor_lstm_units,
            critic_lstm_units=self.ppo.critic_lstm_units,
            actor_mlp_dims=self.ppo.actor_mlp_dims,
            critic_mlp_dims=self.ppo.critic_mlp_dims,
            lstm_layers=self.ppo.lstm_layers,
            lstm_layer_norm=self.ppo.lstm_layer_norm,
            activation=self.ppo.activation,
            init_sigma=self.ppo.init_sigma,
        ).to(self.device)

        # Separate optimizers for actor and critic (DEXTRAH: different LRs)
        self.actor_params = (
            list(self.actor_critic.actor_lstm.parameters())
            + list(self.actor_critic.actor_mlp.parameters())
            + [self.actor_critic.actor_mean.weight, self.actor_critic.actor_mean.bias]
            + [self.actor_critic.actor_log_std]
        )
        self.critic_params = (
            list(self.actor_critic.critic_lstm.parameters())
            + list(self.actor_critic.critic_mlp.parameters())
            + [self.actor_critic.critic_head.weight, self.actor_critic.critic_head.bias]
        )
        self.actor_optimizer = optim.Adam(self.actor_params, lr=self.ppo.learning_rate)
        self.critic_optimizer = optim.Adam(self.critic_params, lr=self.ppo.critic_lr)

        # Rollout buffer
        self.buffer = SequenceRolloutBuffer(
            num_steps=self.ppo.rollout_steps,
            num_envs=env.num_envs,
            num_actor_obs=env.num_teacher_obs,
            num_critic_obs=env.num_teacher_obs,
            num_actions=env.num_actions,
            device=self.device,
        )

        # Observation normalization (DEXTRAH: normalize_input=True with running_mean_std)
        self.obs_normalizer = ObsRunningMeanStd(
            shape=(env.num_teacher_obs,), device=self.device
        ) if self.ppo.normalize_input else None

        # Value normalization
        self.value_normalizer = RunningMeanStd(device=self.device) if self.ppo.normalize_value else None

        # LSTM hidden states (persistent across rollouts)
        self.actor_hidden = None
        self.critic_hidden = None

        # Logging
        self.writer = None

    def collect_rollout(self) -> None:
        """Collect rollout data with LSTM hidden state tracking."""
        for step in range(self.ppo.rollout_steps):
            # Store hidden states for this step
            if self.actor_hidden is not None:
                self.buffer.store_hidden(self.actor_hidden, self.critic_hidden, step)

            # Get teacher observations and normalize
            raw_obs = self.env.teacher_obs_buf
            if self.obs_normalizer is not None:
                self.obs_normalizer.update(raw_obs)
                actor_obs = self.obs_normalizer.normalize(raw_obs)
            else:
                actor_obs = raw_obs
            critic_obs = actor_obs  # Same normalized obs for both

            with torch.no_grad():
                result = self.actor_critic.get_action_and_value(
                    actor_obs=actor_obs,
                    critic_obs=critic_obs,
                    actor_hidden=self.actor_hidden,
                    critic_hidden=self.critic_hidden,
                )

                action = torch.clamp(result["action"], -1.0, 1.0)
                value = result["value"]

                # Denormalize value for GAE computation
                if self.value_normalizer is not None:
                    value = self.value_normalizer.denormalize(value)

                # Update persistent hidden states
                self.actor_hidden = result["actor_hidden"]
                self.critic_hidden = result["critic_hidden"]

            # Step environment
            _, reward, terminated, truncated, _ = self.env.step(action)
            done = terminated | truncated

            # Reset hidden states for done environments
            done_indices = torch.where(done)[0]
            if len(done_indices) > 0 and self.actor_hidden is not None:
                for idx in done_indices:
                    self.actor_hidden[0][:, idx] = 0
                    self.actor_hidden[1][:, idx] = 0
                    self.critic_hidden[0][:, idx] = 0
                    self.critic_hidden[1][:, idx] = 0

            # Store transition
            self.buffer.add(
                actor_obs, critic_obs,
                action, result["log_prob"],
                reward, done, value,
                result["action_mean"],
                timeout=truncated,
            )

            # Track metrics
            self._episode_rewards += reward
            self._episode_lengths += 1
            for idx in done_indices:
                self._completed_episodes += 1
                self._total_episode_reward += self._episode_rewards[idx].item()
                self._total_episode_length += self._episode_lengths[idx].item()
                self._episode_rewards[idx] = 0
                self._episode_lengths[idx] = 0

            # Track reward components
            for key, val in self.env.reward_components.items():
                if key not in self._rc_sums:
                    self._rc_sums[key] = 0.0
                self._rc_sums[key] += val
            self._rc_counts += 1

    def update(self) -> dict[str, float]:
        """Update policy using PPO with sequence-based minibatches."""
        ppo = self.ppo

        # Normalize advantages
        if ppo.normalize_advantage:
            adv = self.buffer.advantages
            self.buffer.advantages = (adv - adv.mean()) / (adv.std() + 1e-8)

        # Snapshot old sigma
        with torch.no_grad():
            old_log_std = torch.clamp(self.actor_critic.actor_log_std, -5.0, 2.0)
            old_sigma = torch.exp(old_log_std)

        metrics = {"policy_loss": 0, "value_loss": 0, "entropy": 0, "bounds_loss": 0, "kl": 0}
        num_updates = 0

        seq_len = min(ppo.seq_len, ppo.rollout_steps)
        num_minibatches = max(1, (ppo.rollout_steps * self.env.num_envs) // ppo.minibatch_size)

        for epoch in range(ppo.num_epochs):
            epoch_kl = 0.0
            epoch_batches = 0

            for batch in self.buffer.get_sequence_batches(seq_len, num_minibatches):
                b_actor_obs = batch["actor_obs"]  # (batch, seq, obs_dim)
                b_critic_obs = batch["critic_obs"]
                b_actions = batch["actions"]
                b_old_log_probs = batch["log_probs"]
                b_advantages = batch["advantages"]
                b_returns = batch["returns"]
                b_old_values = batch["values"]
                b_old_means = batch["action_means"]
                b_dones = batch["dones"]
                b_actor_hidden = batch["actor_hidden"]
                b_critic_hidden = batch["critic_hidden"]

                # Forward pass through LSTM with sequences
                result = self.actor_critic.get_action_and_value(
                    actor_obs=b_actor_obs,
                    critic_obs=b_critic_obs,
                    actor_hidden=b_actor_hidden,
                    critic_hidden=b_critic_hidden,
                    action=b_actions,
                    dones=b_dones,
                )

                new_log_probs = result["log_prob"]
                entropy = result["entropy"]
                new_values = result["value"]
                new_mu = result["action_mean"]
                new_sigma = result["action_std"]

                # Flatten sequence dimension for loss computation
                b_old_log_probs = b_old_log_probs.reshape(-1)
                b_advantages = b_advantages.reshape(-1)
                b_returns = b_returns.reshape(-1)
                b_old_values = b_old_values.reshape(-1)
                b_old_means = b_old_means.reshape(-1, b_old_means.shape[-1])
                new_log_probs = new_log_probs.reshape(-1)
                entropy = entropy.reshape(-1)
                new_values = new_values.reshape(-1)
                new_mu = new_mu.reshape(-1, new_mu.shape[-1])

                # Policy loss
                log_ratio = new_log_probs - b_old_log_probs
                ratio = torch.exp(log_ratio)
                pg_loss1 = -b_advantages * ratio
                pg_loss2 = -b_advantages * torch.clamp(ratio, 1 - ppo.clip_epsilon, 1 + ppo.clip_epsilon)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Value loss (clipped)
                if self.value_normalizer is not None:
                    norm_returns = self.value_normalizer.normalize(b_returns)
                    norm_old_values = self.value_normalizer.normalize(b_old_values)
                else:
                    norm_returns = b_returns
                    norm_old_values = b_old_values

                value_clipped = norm_old_values + torch.clamp(
                    new_values - norm_old_values, -ppo.clip_epsilon, ppo.clip_epsilon
                )
                v_loss1 = (new_values - norm_returns) ** 2
                v_loss2 = (value_clipped - norm_returns) ** 2
                value_loss = 0.5 * torch.max(v_loss1, v_loss2).mean()

                # Entropy
                entropy_loss = entropy.mean()

                # Bounds loss
                soft_bound = 1.1
                mu_high = torch.clamp_min(new_mu - soft_bound, 0.0) ** 2
                mu_low = torch.clamp_max(new_mu + soft_bound, 0.0) ** 2
                bounds_loss = (mu_high + mu_low).sum(dim=-1).mean()

                # Total loss
                actor_loss = pg_loss - ppo.entropy_coef * entropy_loss + ppo.bounds_loss_coef * bounds_loss
                critic_loss = ppo.value_coef * value_loss

                # Combined backward, separate optimizer steps
                self.actor_optimizer.zero_grad()
                self.critic_optimizer.zero_grad()
                (actor_loss + critic_loss).backward()
                nn.utils.clip_grad_norm_(self.actor_params, ppo.max_grad_norm)
                nn.utils.clip_grad_norm_(self.critic_params, ppo.max_grad_norm)
                self.actor_optimizer.step()
                self.critic_optimizer.step()

                # KL divergence
                with torch.no_grad():
                    kl = gaussian_kl(b_old_means, old_sigma.expand_as(b_old_means),
                                     new_mu, new_sigma.expand_as(new_mu) if new_sigma.dim() < new_mu.dim() else new_sigma)

                metrics["policy_loss"] += pg_loss.item()
                metrics["value_loss"] += value_loss.item()
                metrics["entropy"] += entropy_loss.item()
                metrics["bounds_loss"] += bounds_loss.item()
                metrics["kl"] += kl.item()
                num_updates += 1
                epoch_kl += kl.item()
                epoch_batches += 1

            # LR schedule
            avg_epoch_kl = epoch_kl / max(epoch_batches, 1)
            if ppo.lr_schedule == "adaptive":
                current_lr = self.actor_optimizer.param_groups[0]["lr"]
                if avg_epoch_kl > ppo.kl_threshold * 2.0:
                    new_lr = max(current_lr / 1.5, 1e-6)
                elif avg_epoch_kl < ppo.kl_threshold * 0.5:
                    new_lr = min(current_lr * 1.5, 1e-2)
                else:
                    new_lr = current_lr
                for pg in self.actor_optimizer.param_groups:
                    pg["lr"] = new_lr

        for k in metrics:
            metrics[k] /= max(num_updates, 1)
        metrics["learning_rate"] = self.actor_optimizer.param_groups[0]["lr"]
        return metrics

    def train(self):
        """Main training loop."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_name = f"teacher_{self.config.experiment_name}_{timestamp}"
        log_dir = Path("runs") / run_name
        checkpoint_dir = Path(self.config.checkpoint_dir) / run_name

        log_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self.writer = SummaryWriter(log_dir)

        # W&B
        if self.config.wandb.enabled:
            try:
                import wandb
                wandb.init(
                    project=self.config.wandb.project,
                    entity=self.config.wandb.entity,
                    name=run_name,
                    config=asdict(self.config),
                    tags=self.config.wandb.tags + ["teacher"],
                    mode=self.config.wandb.mode,
                )
            except ImportError:
                self.config.wandb.enabled = False

        # Initialize
        self.env.reset()
        self.actor_hidden = self.actor_critic.init_hidden(self.env.num_envs, self.device)["actor"]
        self.critic_hidden = self.actor_critic.init_hidden(self.env.num_envs, self.device)["critic"]

        global_step = 0
        num_updates = 0
        start_time = time.time()
        batch_size = self.ppo.rollout_steps * self.env.num_envs
        best_reward = float("-inf")

        self._episode_rewards = torch.zeros(self.env.num_envs, device=self.device)
        self._episode_lengths = torch.zeros(self.env.num_envs, device=self.device, dtype=torch.int32)
        self._completed_episodes = 0
        self._total_episode_reward = 0.0
        self._total_episode_length = 0
        self._last_avg_reward = 0.0
        self._last_avg_length = 0.0
        self._rc_sums: dict[str, float] = {}
        self._rc_counts = 0

        print("=" * 60)
        print("DEXTRAH-G Teacher Training (Privileged RL with LSTM)")
        print("=" * 60)
        print(f"  Envs: {self.env.num_envs}")
        print(f"  Actor obs: {self.env.num_teacher_obs}D")
        print(f"  Actions: {self.env.num_actions}D")
        print(f"  Actor LSTM: {self.ppo.actor_lstm_units}")
        print(f"  Critic LSTM: {self.ppo.critic_lstm_units}")
        print(f"  LR: actor={self.ppo.learning_rate}, critic={self.ppo.critic_lr}")
        print(f"  Gamma: {self.ppo.gamma}")
        print(f"  Max iterations: {self.ppo.max_iterations}")
        print(f"  Checkpoints: {checkpoint_dir}")
        print("=" * 60)

        for iteration in range(1, self.ppo.max_iterations + 1):
            # Collect rollout
            self.collect_rollout()

            # Compute returns
            with torch.no_grad():
                last_obs = self.env.teacher_obs_buf
                if self.obs_normalizer is not None:
                    last_obs = self.obs_normalizer.normalize(last_obs)
                last_value, self.critic_hidden = self.actor_critic.get_value(
                    last_obs, self.critic_hidden
                )
                if self.value_normalizer is not None:
                    last_value = self.value_normalizer.denormalize(last_value)

            self.buffer.compute_returns_and_advantages(
                last_value, self.ppo.gamma, self.ppo.gae_lambda
            )

            if self.value_normalizer is not None:
                self.value_normalizer.update(self.buffer.returns)

            # Update
            update_metrics = self.update()

            # Linear LR schedule: decay to 0 over max_iterations
            if self.ppo.lr_schedule == "linear":
                progress = iteration / self.ppo.max_iterations
                new_actor_lr = self.ppo.learning_rate * (1.0 - progress)
                new_critic_lr = self.ppo.critic_lr * (1.0 - progress)
                for pg in self.actor_optimizer.param_groups:
                    pg["lr"] = new_actor_lr
                for pg in self.critic_optimizer.param_groups:
                    pg["lr"] = new_critic_lr

            self.buffer.reset()
            global_step += batch_size
            num_updates += 1

            # Logging
            if num_updates % self.config.log_interval == 0:
                elapsed = time.time() - start_time
                fps = global_step / elapsed

                if self._completed_episodes > 0:
                    avg_reward = self._total_episode_reward / self._completed_episodes
                    avg_length = self._total_episode_length / self._completed_episodes
                    self._last_avg_reward = avg_reward
                    self._last_avg_length = avg_length
                else:
                    avg_reward = self._last_avg_reward
                    avg_length = self._last_avg_length

                # Compute average reward components
                avg_rc = {}
                if self._rc_counts > 0:
                    for k, v in self._rc_sums.items():
                        avg_rc[k] = v / self._rc_counts

                # Console output: main metrics
                print(
                    f"[iter {iteration:>6} | step {global_step:>10}] "
                    f"FPS: {fps:.0f} | "
                    f"reward: {avg_reward:.3f} | "
                    f"len: {avg_length:.1f} | "
                    f"ep: {self._completed_episodes} | "
                    f"pg: {update_metrics['policy_loss']:.4f} | "
                    f"vl: {update_metrics['value_loss']:.4f} | "
                    f"kl: {update_metrics['kl']:.4f} | "
                    f"lr: {update_metrics['learning_rate']:.2e}"
                )
                # Console output: reward components
                if avg_rc:
                    h2o = avg_rc.get("hand_to_object_reward", 0)
                    o2g = avg_rc.get("object_to_goal_reward", 0)
                    lift = avg_rc.get("lift_reward", 0)
                    curl = avg_rc.get("finger_curl_reg", 0)
                    h2o_d = avg_rc.get("hand_to_object_dist", 0)
                    o2g_d = avg_rc.get("object_to_goal_dist", 0)
                    c_h = avg_rc.get("cube_height", 0)
                    p_r = avg_rc.get("phase_reach", 0)
                    p_g = avg_rc.get("phase_grasp", 0)
                    p_l = avg_rc.get("phase_lift", 0)
                    print(
                        f"  rewards  | h2o: {h2o:.3f} | o2g: {o2g:.3f} | "
                        f"lift: {lift:.3f} | curl: {curl:.4f}"
                    )
                    print(
                        f"  metrics  | h2o_d: {h2o_d:.3f} | o2g_d: {o2g_d:.3f} | "
                        f"cube_h: {c_h:.3f} | "
                        f"phase: R{p_r:.0%}/G{p_g:.0%}/L{p_l:.0%}"
                    )

                if self.writer:
                    self.writer.add_scalar("charts/fps", fps, global_step)
                    self.writer.add_scalar("charts/learning_rate", update_metrics["learning_rate"], global_step)
                    self.writer.add_scalar("episode/reward_mean", avg_reward, global_step)
                    self.writer.add_scalar("episode/length_mean", avg_length, global_step)
                    for k, v in update_metrics.items():
                        self.writer.add_scalar(f"losses/{k}", v, global_step)
                    for k, v in avg_rc.items():
                        self.writer.add_scalar(f"reward/{k}", v, global_step)

                if self.config.wandb.enabled:
                    import wandb
                    wandb.log({
                        "charts/fps": fps,
                        "episode/reward_mean": avg_reward,
                        "episode/length_mean": avg_length,
                        **{f"losses/{k}": v for k, v in update_metrics.items()},
                        **{f"reward/{k}": v for k, v in avg_rc.items()},
                    }, step=global_step)

                # Reset tracking
                self._completed_episodes = 0
                self._total_episode_reward = 0.0
                self._total_episode_length = 0
                self._rc_sums = {}
                self._rc_counts = 0

            # Save best
            if self._completed_episodes > 0:
                current_reward = self._total_episode_reward / self._completed_episodes
                if current_reward > best_reward:
                    best_reward = current_reward
                    self.save(checkpoint_dir / "best.pt")

            # Save periodic checkpoint
            if num_updates % self.config.save_interval == 0:
                self.save(checkpoint_dir / f"checkpoint_{global_step}.pt")

        # Final save
        self.save(checkpoint_dir / "final.pt")
        print(f"[INFO] Training complete! Final model: {checkpoint_dir / 'final.pt'}")

        if self.writer:
            self.writer.close()
        if self.config.wandb.enabled:
            import wandb
            wandb.finish()

    def save(self, path: str | Path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "actor_critic": self.actor_critic.state_dict(),
            "actor_optimizer": self.actor_optimizer.state_dict(),
            "critic_optimizer": self.critic_optimizer.state_dict(),
            "config": asdict(self.config),
        }, path)

    def load(self, path: str | Path):
        ckpt = torch.load(path, map_location=self.device)
        self.actor_critic.load_state_dict(ckpt["actor_critic"])
        if "actor_optimizer" in ckpt:
            self.actor_optimizer.load_state_dict(ckpt["actor_optimizer"])
        if "critic_optimizer" in ckpt:
            self.critic_optimizer.load_state_dict(ckpt["critic_optimizer"])


def main():
    parser = argparse.ArgumentParser(description="DEXTRAH-G Teacher Training")

    parser.add_argument("--num-envs", type=int, default=256)
    parser.add_argument("--max-iterations", type=int, default=20000)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-wandb", action="store_true")
    parser.add_argument("--wandb-project", type=str, default="newton-franka-allegro-grasp")
    parser.add_argument("--log-interval", type=int, default=10)
    parser.add_argument("--save-interval", type=int, default=100)
    parser.add_argument("--checkpoint", type=str, default=None)

    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    env_config = EnvConfig(
        num_envs=args.num_envs,
        use_depth_sensor=False,  # Teacher doesn't use depth
        use_fabric_actions=True,
    )

    teacher_config = TeacherPPOConfig(
        max_iterations=args.max_iterations,
        learning_rate=args.learning_rate,
    )

    train_config = TrainConfig(
        seed=args.seed,
        log_interval=args.log_interval,
        save_interval=args.save_interval,
        env=env_config,
        teacher=teacher_config,
    )
    train_config.wandb.enabled = not args.no_wandb
    train_config.wandb.project = args.wandb_project

    print("[INFO] Creating environment...")
    env = FrankaAllegroGraspEnv(env_config, device="cuda", headless=True)

    print("[INFO] Creating teacher trainer...")
    trainer = TeacherTrainer(env=env, config=train_config)

    if args.checkpoint:
        print(f"[INFO] Loading checkpoint: {args.checkpoint}")
        trainer.load(args.checkpoint)

    print("[INFO] Starting teacher training...")
    trainer.train()


if __name__ == "__main__":
    main()
