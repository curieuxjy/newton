"""Training script for Allegro Hand Cube Rotation with PPO."""

import os
import time
from dataclasses import asdict
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

    # Flatten config for wandb
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
        "reward_rotation": config.env.reward_rotation,
        "reward_action_penalty": config.env.reward_action_penalty,
        # PPO config
        "learning_rate": config.ppo.learning_rate,
        "gamma": config.ppo.gamma,
        "gae_lambda": config.ppo.gae_lambda,
        "clip_epsilon": config.ppo.clip_epsilon,
        "entropy_coef": config.ppo.entropy_coef,
        "value_coef": config.ppo.value_coef,
        "num_epochs": config.ppo.num_epochs,
        "num_minibatches": config.ppo.num_minibatches,
        "rollout_steps": config.ppo.rollout_steps,
        "total_timesteps": config.ppo.total_timesteps,
        "hidden_dims": config.ppo.hidden_dims,
    }

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

    # Watch model for gradient tracking
    wandb.watch(actor_critic, log="gradients", log_freq=100)

    print(f"[INFO] wandb initialized: {wandb.run.url}")
    return True


def train(config: TrainConfig):
    """Main training loop."""
    # Set random seeds
    torch.manual_seed(config.seed)

    device = config.device
    print(f"[INFO] Using device: {device}")

    # Create environment
    print("[INFO] Creating environment...")
    env = AllegroHandCubeEnv(config.env, device=device, headless=True)
    print(f"[INFO] Num envs: {env.num_envs}")
    print(f"[INFO] Obs dim: {env.num_obs}, Action dim: {env.num_actions}")

    # Create actor-critic network
    actor_critic = ActorCritic(
        num_obs=env.num_obs,
        num_actions=env.num_actions,
        hidden_dims=config.ppo.hidden_dims,
        activation=config.ppo.activation,
    )
    print(f"[INFO] Actor-Critic parameters: {sum(p.numel() for p in actor_critic.parameters()):,}")

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
    )

    # Create checkpoint directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    checkpoint_dir = os.path.join(config.checkpoint_dir, f"{config.experiment_name}_{timestamp}")
    os.makedirs(checkpoint_dir, exist_ok=True)
    print(f"[INFO] Checkpoints will be saved to: {checkpoint_dir}")

    # Training loop
    num_updates = config.ppo.total_timesteps // (config.ppo.rollout_steps * env.num_envs)
    print(f"[INFO] Total updates: {num_updates}")

    # Initialize environment
    obs = env.reset()
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

    for update in range(1, num_updates + 1):
        update_start = time.time()

        # Collect rollout
        buffer.reset()
        for step in range(config.ppo.rollout_steps):
            with torch.no_grad():
                action, log_prob, _, value = actor_critic.get_action_and_value(obs)

            # Environment step
            next_obs, reward, done, info = env.step(action)
            global_step += env.num_envs

            # Store in buffer
            buffer.add(obs, action, log_prob, reward, done, value)

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

            obs = next_obs

        # Compute returns and advantages
        with torch.no_grad():
            last_value = actor_critic.get_value(obs)
        buffer.compute_returns_and_advantages(last_value, config.ppo.gamma, config.ppo.gae_lambda)

        # PPO update
        metrics = ppo.update(buffer)

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
                f"Ent {metrics['entropy']:6.3f}"
            )

            # TensorBoard logging
            if writer:
                writer.add_scalar("train/policy_loss", metrics["policy_loss"], global_step)
                writer.add_scalar("train/value_loss", metrics["value_loss"], global_step)
                writer.add_scalar("train/entropy", metrics["entropy"], global_step)
                writer.add_scalar("episode/reward_mean", avg_reward, global_step)
                writer.add_scalar("episode/length_mean", avg_length, global_step)
                writer.add_scalar("perf/fps", fps, global_step)

            # Wandb logging
            if use_wandb:
                wandb.log(
                    {
                        # Training metrics
                        "train/policy_loss": metrics["policy_loss"],
                        "train/value_loss": metrics["value_loss"],
                        "train/entropy": metrics["entropy"],
                        "train/total_loss": metrics["total_loss"],
                        # Episode metrics
                        "episode/reward_mean": avg_reward,
                        "episode/length_mean": avg_length,
                        "episode/completed": completed_episodes,
                        # Performance
                        "perf/fps": fps,
                        "perf/global_step": global_step,
                        "perf/update": update,
                    },
                    step=global_step,
                )

            # Reset episode tracking
            completed_episodes = 0
            total_episode_reward = 0.0
            total_episode_length = 0

        # Save checkpoint
        if update % config.save_interval == 0:
            checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_{update}.pt")
            ppo.save(checkpoint_path)
            print(f"[INFO] Saved checkpoint: {checkpoint_path}")

            # Save to wandb
            if use_wandb and config.wandb.save_model:
                wandb.save(checkpoint_path)

    # Final save
    final_path = os.path.join(checkpoint_dir, "final.pt")
    ppo.save(final_path)
    print(f"[INFO] Training complete! Final model saved to: {final_path}")

    # Finalize wandb
    if use_wandb:
        # Save final model as artifact
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

    # Close TensorBoard
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
    # Wandb arguments
    parser.add_argument("--wandb", action="store_true", default=True, help="Enable wandb logging")
    parser.add_argument("--no-wandb", action="store_false", dest="wandb", help="Disable wandb logging")
    parser.add_argument("--wandb-project", type=str, default="allegro-cube-ppo", help="Wandb project name")
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

    # Wandb config
    config.wandb.enabled = args.wandb
    config.wandb.project = args.wandb_project
    config.wandb.entity = args.wandb_entity
    config.wandb.group = args.wandb_group

    train(config)


if __name__ == "__main__":
    main()
