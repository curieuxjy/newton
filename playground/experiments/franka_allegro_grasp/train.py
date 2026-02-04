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
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Get action, log prob, entropy, and value."""
        # Sanitize input
        state_obs = torch.nan_to_num(state_obs, nan=0.0, posinf=10.0, neginf=-10.0)
        state_obs = torch.clamp(state_obs, -100.0, 100.0)

        if fabric_obs is not None:
            fabric_obs = torch.nan_to_num(fabric_obs, nan=0.0, posinf=10.0, neginf=-10.0)
            fabric_obs = torch.clamp(fabric_obs, -100.0, 100.0)

        combined = self._encode_features(state_obs, fabric_obs)
        features = self.shared(combined)
        action_mean = self.actor_mean(features)

        # Clamp log_std
        log_std_clamped = torch.clamp(self.actor_log_std, min=-20.0, max=2.0)
        action_std = torch.exp(log_std_clamped)

        action_mean = torch.nan_to_num(action_mean, nan=0.0, posinf=1.0, neginf=-1.0)

        dist = Normal(action_mean, action_std)

        if action is None:
            action = dist.sample()

        log_prob = dist.log_prob(action).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)
        value = self.critic(features).squeeze(-1)

        return action, log_prob, entropy, value

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
    ):
        """Add a step of data."""
        self.state_observations[self.step] = state_obs
        if fabric_obs is not None and self.fabric_observations is not None:
            self.fabric_observations[self.step] = fabric_obs
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
        log_probs_flat = self.log_probs.reshape(-1)
        advantages_flat = self.advantages.reshape(-1)
        returns_flat = self.returns.reshape(-1)

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
                action, log_prob, _, value = self.actor_critic.get_action_and_value(
                    state_obs[:, : self.env.num_state_obs], fabric_obs
                )
                action = torch.clamp(action, -1.0, 1.0)

            # Step environment
            next_obs, reward, done, info = self.env.step(action)

            # Store in buffer
            self.buffer.add(
                state_obs[:, : self.env.num_state_obs],
                fabric_obs,
                action,
                log_prob,
                reward,
                done,
                value,
            )

            state_obs = next_obs

        return state_obs

    def update(self) -> dict[str, float]:
        """Update policy using PPO."""
        # Normalize advantages
        advantages = self.buffer.advantages.reshape(-1)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        self.buffer.advantages = advantages.reshape(
            self.buffer.num_steps, self.buffer.num_envs
        )

        metrics = {
            "policy_loss": 0.0,
            "value_loss": 0.0,
            "entropy": 0.0,
            "total_loss": 0.0,
        }
        num_updates = 0

        for _ in range(self.config.ppo.num_epochs):
            for batch in self.buffer.get_batches(self.config.ppo.num_minibatches):
                state_obs, fabric_obs, actions, old_log_probs, advs, returns = batch

                # Get current policy outputs
                _, new_log_probs, entropy, new_values = self.actor_critic.get_action_and_value(
                    state_obs, fabric_obs, actions
                )

                # Policy loss
                log_ratio = new_log_probs - old_log_probs
                ratio = torch.exp(log_ratio)

                pg_loss1 = -advs * ratio
                pg_loss2 = -advs * torch.clamp(
                    ratio,
                    1 - self.config.ppo.clip_epsilon,
                    1 + self.config.ppo.clip_epsilon,
                )
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Value loss
                value_loss = 0.5 * ((new_values - returns) ** 2).mean()

                # Entropy loss
                entropy_loss = entropy.mean()

                # Total loss
                loss = (
                    pg_loss
                    + self.config.ppo.value_coef * value_loss
                    - self.config.ppo.entropy_coef * entropy_loss
                )

                # Gradient step
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(
                    self.actor_critic.parameters(), self.config.ppo.max_grad_norm
                )
                self.optimizer.step()

                metrics["policy_loss"] += pg_loss.item()
                metrics["value_loss"] += value_loss.item()
                metrics["entropy"] += entropy_loss.item()
                metrics["total_loss"] += loss.item()
                num_updates += 1

        for key in metrics:
            metrics[key] /= num_updates

        return metrics

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

        print(f"[INFO] Starting training: {total_timesteps} timesteps")
        print(f"[INFO] Batch size: {batch_size}")
        print(f"[INFO] Checkpoints: {checkpoint_dir}")

        while global_step < total_timesteps:
            # Collect rollout
            state_obs = self.collect_rollout(state_obs)

            # Compute returns and advantages
            with torch.no_grad():
                fabric_obs = self.compute_fabric_obs()
                last_value = self.actor_critic.get_value(
                    state_obs[:, : self.env.num_state_obs], fabric_obs
                )

            self.buffer.compute_returns_and_advantages(
                last_value, self.config.ppo.gamma, self.config.ppo.gae_lambda
            )

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

                # Get reward info
                reward_components = self.env.reward_components
                avg_reward = self.buffer.rewards.mean().item() if self.buffer.step > 0 else 0

                print(
                    f"[{global_step:>10}] "
                    f"FPS: {fps:.0f} | "
                    f"reward: {avg_reward:.3f} | "
                    f"policy_loss: {update_metrics['policy_loss']:.4f} | "
                    f"value_loss: {update_metrics['value_loss']:.4f}"
                )

                # TensorBoard
                self.writer.add_scalar("charts/fps", fps, global_step)
                self.writer.add_scalar("losses/policy_loss", update_metrics["policy_loss"], global_step)
                self.writer.add_scalar("losses/value_loss", update_metrics["value_loss"], global_step)
                self.writer.add_scalar("losses/entropy", update_metrics["entropy"], global_step)

                for key, value in reward_components.items():
                    self.writer.add_scalar(f"reward/{key}", value, global_step)

                # Wandb
                if self.config.wandb.enabled:
                    wandb.log(
                        {
                            "fps": fps,
                            "global_step": global_step,
                            **{f"losses/{k}": v for k, v in update_metrics.items()},
                            **{f"reward/{k}": v for k, v in reward_components.items()},
                        }
                    )

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

    # Training
    parser.add_argument("--total-timesteps", type=int, default=10_000_000, help="Total timesteps")
    parser.add_argument("--learning-rate", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--rollout-steps", type=int, default=24, help="Rollout steps per update")
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
