"""Training script for Allegro Hand Cube Rotation with PPO."""

import os
import time
from datetime import datetime

import torch

from .config import TrainConfig
from .env import AllegroHandCubeEnv
from .ppo import ActorCritic, PPO, RolloutBuffer

# Optional wandb import
try:
    import wandb

    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    wandb = None

# TensorBoard import
try:
    from torch.utils.tensorboard import SummaryWriter

    TENSORBOARD_AVAILABLE = True
except ImportError:
    TENSORBOARD_AVAILABLE = False
    SummaryWriter = None


def init_wandb(config: TrainConfig, actor_critic: ActorCritic) -> bool:
    """Initialize Weights & Biases logging."""
    if not config.wandb.enabled or not WANDB_AVAILABLE:
        if config.wandb.enabled and not WANDB_AVAILABLE:
            print("[WARN] wandb not installed. Run: pip install wandb")
        return False

    wandb_config = {
        "seed": config.seed,
        "device": config.device,
        # Env config
        "num_envs": config.env.num_envs,
        "episode_length": config.env.episode_length,
        "fps": config.env.fps,
        "sim_substeps": config.env.sim_substeps,
        "control_decimation": config.env.control_decimation,
        "hand_stiffness": config.env.hand_stiffness,
        "hand_damping": config.env.hand_damping,
        "rot_reward_scale": config.env.rot_reward_scale,
        "action_penalty_scale": config.env.action_penalty_scale,
        # PPO config
        "network_type": config.ppo.network_type,
        "learning_rate": config.ppo.learning_rate,
        "critic_learning_rate": config.ppo.critic_learning_rate,
        "lr_schedule": config.ppo.lr_schedule,
        "gamma": config.ppo.gamma,
        "gae_lambda": config.ppo.gae_lambda,
        "clip_epsilon": config.ppo.clip_epsilon,
        "entropy_coef": config.ppo.entropy_coef,
        "value_coef": config.ppo.value_coef,
        "num_epochs": config.ppo.num_epochs,
        "minibatch_size": config.ppo.minibatch_size,
        "rollout_steps": config.ppo.rollout_steps,
        "total_timesteps": config.ppo.total_timesteps,
        "use_layer_norm": config.ppo.use_layer_norm,
        # DR config
        "dr_mode": config.dr.mode,
    }
    if config.dr.mode == "adr":
        wandb_config.update({
            "adr_boundary_fraction": config.dr.adr_boundary_fraction,
            "adr_queue_length": config.dr.adr_queue_length,
            "adr_threshold_high": config.dr.adr_threshold_high,
            "adr_threshold_low": config.dr.adr_threshold_low,
        })
    # Architecture-specific config
    if config.ppo.network_type == "lstm":
        wandb_config.update(
            {
                "actor_lstm_size": config.ppo.actor_lstm_size,
                "critic_lstm_size": config.ppo.critic_lstm_size,
                "actor_mlp_dims": config.ppo.actor_mlp_dims,
                "critic_mlp_dims": config.ppo.critic_mlp_dims,
            }
        )
    else:
        wandb_config["mlp_hidden_dims"] = config.ppo.mlp_hidden_dims

    wandb.init(
        project=config.wandb.project,
        entity=config.wandb.entity,
        group=config.wandb.group,
        tags=config.wandb.tags,
        name=f"{config.experiment_name}_{datetime.now().strftime('%m%d_%H%M')}",
        config=wandb_config,
        mode=config.wandb.mode,
        save_code=True,
    )

    wandb.watch(actor_critic, log="gradients", log_freq=100)

    print(f"[INFO] wandb initialized: {wandb.run.url}")
    return True


def train(config: TrainConfig):
    """Main training loop."""
    torch.manual_seed(config.seed)

    device = config.device
    print(f"[INFO] Using device: {device}")

    # Create environment
    print("[INFO] Creating environment...")
    env = AllegroHandCubeEnv(config.env, device=device, headless=True, dr_config=config.dr)
    print(f"[INFO] Num envs: {env.num_envs}")
    print(f"[INFO] Obs dim: {env.num_obs}, Action dim: {env.num_actions}")

    # Create actor-critic network
    actor_critic = ActorCritic(
        num_obs=env.num_obs,
        num_actions=env.num_actions,
        config=config.ppo,
    )
    num_params = sum(p.numel() for p in actor_critic.parameters())
    num_actor = sum(p.numel() for p in actor_critic.actor_parameters())
    num_critic = sum(p.numel() for p in actor_critic.critic_parameters())
    print(f"[INFO] Network type: {config.ppo.network_type}")
    print(f"[INFO] Total parameters: {num_params:,} (actor: {num_actor:,}, critic: {num_critic:,})")

    # Create PPO algorithm
    ppo = PPO(actor_critic, config.ppo, device=device)

    # Initialize wandb
    use_wandb = init_wandb(config, actor_critic)

    # Initialize TensorBoard
    writer = None
    if TENSORBOARD_AVAILABLE:
        log_dir = os.path.join("runs", f"{config.experiment_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
        writer = SummaryWriter(log_dir)
        print(f"[INFO] TensorBoard logging to: {log_dir}")
        print(f"[INFO] Run: tensorboard --logdir runs/")

    # Create rollout buffer
    buffer = RolloutBuffer(
        num_steps=config.ppo.rollout_steps,
        num_envs=env.num_envs,
        num_obs=env.num_obs,
        num_actions=env.num_actions,
        device=device,
        actor_lstm_size=config.ppo.actor_lstm_size if config.ppo.network_type == "lstm" else 0,
        critic_lstm_size=config.ppo.critic_lstm_size if config.ppo.network_type == "lstm" else 0,
    )

    # Create checkpoint directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    checkpoint_dir = os.path.join(config.checkpoint_dir, f"{config.experiment_name}_{timestamp}")
    os.makedirs(checkpoint_dir, exist_ok=True)
    print(f"[INFO] Checkpoints will be saved to: {checkpoint_dir}")

    # Training loop
    num_updates = config.ppo.total_timesteps // (config.ppo.rollout_steps * env.num_envs)
    print(f"[INFO] Total updates: {num_updates}")

    # Initialize environment and LSTM hidden states
    obs = env.reset()
    hidden = actor_critic.init_hidden(env.num_envs, device)  # None for MLP
    global_step = 0
    start_time = time.time()

    # Metrics tracking
    episode_rewards = torch.zeros(env.num_envs, device=device)
    episode_lengths = torch.zeros(env.num_envs, device=device, dtype=torch.int32)
    completed_episodes = 0
    total_episode_reward = 0.0
    total_episode_length = 0

    print("[INFO] Starting training...")
    print("-" * 60)

    # Reward component tracking
    reward_component_sums = {}
    reward_component_counts = 0

    for update in range(1, num_updates + 1):
        update_start = time.time()

        # Store initial hidden state for LSTM training
        buffer.reset()
        if actor_critic.is_lstm:
            buffer.set_initial_hidden(hidden)

        # Collect rollout
        for step in range(config.ppo.rollout_steps):
            with torch.no_grad():
                action, log_prob, _, value, action_mean, _, hidden = actor_critic.get_action_and_value(
                    obs, hidden=hidden
                )
                action = torch.clamp(action, -1.0, 1.0)
                # Denormalize value (critic predicts in normalized space)
                if ppo.value_normalizer is not None:
                    value = ppo.value_normalizer.denormalize(value)

            # Environment step
            next_obs, reward, done, info = env.step(action)
            global_step += env.num_envs

            # Store in buffer
            buffer.add(obs, action, log_prob, reward, done, value, action_mean)

            # Track reward components
            if hasattr(env, "reward_components"):
                for key, val in env.reward_components.items():
                    if key not in reward_component_sums:
                        reward_component_sums[key] = 0.0
                    reward_component_sums[key] += val
                reward_component_counts += 1

            # Track episode metrics
            episode_rewards += reward
            episode_lengths += 1

            # Handle episode completions
            done_indices = torch.where(done)[0]
            for idx in done_indices:
                completed_episodes += 1
                total_episode_reward += episode_rewards[idx].item()
                total_episode_length += episode_lengths[idx].item()
                episode_rewards[idx] = 0
                episode_lengths[idx] = 0

            # Reset LSTM hidden states for done environments
            if hidden is not None and len(done_indices) > 0:
                hidden = actor_critic.reset_hidden(hidden, done_indices)

            obs = next_obs

        # Compute returns and advantages
        with torch.no_grad():
            last_value, _ = actor_critic.get_value(obs, hidden=hidden)
            if ppo.value_normalizer is not None:
                last_value = ppo.value_normalizer.denormalize(last_value)
        buffer.compute_returns_and_advantages(last_value, config.ppo.gamma, config.ppo.gae_lambda)

        # Update value normalizer with computed returns
        if ppo.value_normalizer is not None:
            ppo.value_normalizer.update(buffer.returns)

        # PPO update
        metrics = ppo.update(buffer, update_num=update, total_updates=num_updates)

        # Logging
        if update % config.log_interval == 0:
            elapsed = time.time() - start_time
            fps = global_step / elapsed

            avg_reward = total_episode_reward / max(completed_episodes, 1)
            avg_length = total_episode_length / max(completed_episodes, 1)

            print(
                f"Update {update:6d} | "
                f"Steps {global_step:10d} | "
                f"FPS {fps:8.0f} | "
                f"Reward {avg_reward:8.2f} | "
                f"Length {avg_length:6.1f} | "
                f"PL {metrics['policy_loss']:7.4f} | "
                f"VL {metrics['value_loss']:7.4f} | "
                f"Ent {metrics['entropy']:6.3f} | "
                f"KL {metrics['kl_divergence']:.4f} | "
                f"aLR {metrics['actor_learning_rate']:.2e} | "
                f"cLR {metrics['critic_learning_rate']:.2e}"
            )

            # Compute average reward components
            avg_reward_components = {}
            if reward_component_counts > 0:
                for key, val in reward_component_sums.items():
                    avg_reward_components[key] = val / reward_component_counts

            # Print reward components
            if avg_reward_components:
                rc = avg_reward_components
                print(
                    f"  Rewards: rot={rc.get('rot_rew', 0):.2f} "
                    f"dist={rc.get('dist_rew', 0):.2f} "
                    f"act={rc.get('action_penalty', 0):.3f} "
                    f"vel={rc.get('velocity_penalty', 0):.3f} "
                    f"goal={rc.get('reach_goal_rew', 0):.2f} "
                    f"| rot_dist={rc.get('rot_dist', 0):.3f} "
                    f"goal_dist={rc.get('goal_dist', 0):.3f} "
                    f"successes={rc.get('consecutive_successes', 0):.1f}"
                )

            # Split reward components: adr/ keys → randomization/, rest → reward/
            reward_logs = {}
            adr_logs = {}
            for key, val in avg_reward_components.items():
                if key.startswith("adr/"):
                    adr_logs[f"randomization/{key}"] = val
                else:
                    reward_logs[f"reward/{key}"] = val

            # TensorBoard logging
            if writer:
                writer.add_scalar("train/policy_loss", metrics["policy_loss"], global_step)
                writer.add_scalar("train/value_loss", metrics["value_loss"], global_step)
                writer.add_scalar("train/entropy", metrics["entropy"], global_step)
                writer.add_scalar("train/kl_divergence", metrics["kl_divergence"], global_step)
                writer.add_scalar("train/actor_learning_rate", metrics["actor_learning_rate"], global_step)
                writer.add_scalar("train/critic_learning_rate", metrics["critic_learning_rate"], global_step)
                writer.add_scalar("episode/reward_mean", avg_reward, global_step)
                writer.add_scalar("episode/length_mean", avg_length, global_step)
                writer.add_scalar("perf/fps", fps, global_step)
                for key, val in reward_logs.items():
                    writer.add_scalar(key, val, global_step)
                for key, val in adr_logs.items():
                    writer.add_scalar(key, val, global_step)

            # Wandb logging
            if use_wandb:
                log_dict = {
                    "train/policy_loss": metrics["policy_loss"],
                    "train/value_loss": metrics["value_loss"],
                    "train/entropy": metrics["entropy"],
                    "train/total_loss": metrics["total_loss"],
                    "train/kl_divergence": metrics["kl_divergence"],
                    "train/actor_learning_rate": metrics["actor_learning_rate"],
                    "train/critic_learning_rate": metrics["critic_learning_rate"],
                    "episode/reward_mean": avg_reward,
                    "episode/length_mean": avg_length,
                    "perf/fps": fps,
                    "perf/global_step": global_step,
                    "perf/update": update,
                }
                log_dict.update(reward_logs)
                log_dict.update(adr_logs)
                wandb.log(log_dict, step=global_step)

            # Reset episode tracking
            completed_episodes = 0
            total_episode_reward = 0.0
            total_episode_length = 0

            # Reset reward component tracking
            reward_component_sums = {}
            reward_component_counts = 0

        # Save checkpoint
        if update % config.save_interval == 0:
            checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_{update}.pt")
            ppo.save(checkpoint_path)

            # Save ADR state alongside PPO checkpoint
            if env.adr is not None:
                adr_path = os.path.join(checkpoint_dir, f"adr_{update}.pt")
                torch.save(env.adr.state_dict(), adr_path)
                print(f"[INFO] Saved checkpoint: {checkpoint_path} + ADR: {adr_path}")
            else:
                print(f"[INFO] Saved checkpoint: {checkpoint_path}")

            if use_wandb and config.wandb.save_model:
                wandb.save(checkpoint_path)

    # Final save
    final_path = os.path.join(checkpoint_dir, "final.pt")
    ppo.save(final_path)
    if env.adr is not None:
        adr_final_path = os.path.join(checkpoint_dir, "adr_final.pt")
        torch.save(env.adr.state_dict(), adr_final_path)
    print(f"[INFO] Training complete! Final model saved to: {final_path}")

    # Finalize wandb
    if use_wandb:
        if config.wandb.save_model:
            artifact = wandb.Artifact(
                name=f"{config.experiment_name}_model",
                type="model",
                description="Final trained PPO model",
            )
            artifact.add_file(final_path)
            wandb.log_artifact(artifact)

        wandb.finish()
        print("[INFO] wandb run finished")

    if writer:
        writer.close()

    env.close()


def main():
    """Entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="Train Allegro Hand Cube Rotation with PPO")
    parser.add_argument("--num-envs", type=int, default=4096, help="Number of parallel environments")
    parser.add_argument("--total-timesteps", type=int, default=100_000_000, help="Total training timesteps")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--device", type=str, default="cuda", help="Device (cuda or cpu)")
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints", help="Checkpoint directory")
    parser.add_argument(
        "--network-type", type=str, default="lstm", choices=["lstm", "mlp"], help="Network type: lstm (paper) or mlp"
    )
    # DR arguments
    parser.add_argument(
        "--dr-mode", type=str, default="adr", choices=["off", "static", "adr"],
        help="Domain randomisation mode: off, static, or adr (vectorised ADR)"
    )
    # Wandb arguments
    parser.add_argument("--wandb", action="store_true", default=True, help="Enable wandb logging")
    parser.add_argument("--no-wandb", action="store_false", dest="wandb", help="Disable wandb logging")
    parser.add_argument("--wandb-project", type=str, default="newton-allegro-cube", help="Wandb project name")
    parser.add_argument("--wandb-entity", type=str, default=None, help="Wandb entity (username or team)")
    parser.add_argument("--wandb-group", type=str, default=None, help="Wandb run group")
    args = parser.parse_args()

    # Create config
    config = TrainConfig(
        seed=args.seed,
        device=args.device,
        checkpoint_dir=args.checkpoint_dir,
    )
    config.env.num_envs = args.num_envs
    config.ppo.total_timesteps = args.total_timesteps
    config.ppo.network_type = args.network_type

    # DR config
    config.dr.mode = args.dr_mode

    # Wandb config
    config.wandb.enabled = args.wandb
    config.wandb.project = args.wandb_project
    config.wandb.entity = args.wandb_entity
    config.wandb.group = args.wandb_group

    train(config)


if __name__ == "__main__":
    main()
