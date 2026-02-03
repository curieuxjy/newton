"""Hyperparameters for Allegro Hand Cube Rotation PPO training."""

from dataclasses import dataclass, field


@dataclass
class EnvConfig:
    """Environment configuration."""

    num_envs: int = 4096
    episode_length: int = 600  # 12 seconds at 50Hz

    # Simulation
    fps: int = 50
    sim_substeps: int = 4
    control_decimation: int = 2  # control at 25Hz

    # Robot (matching original allegro example)
    hand_stiffness: float = 150.0
    hand_damping: float = 5.0

    # Cube
    cube_size: float = 0.065  # 6.5cm cube
    cube_mass: float = 0.1  # 100g

    # Reward weights
    reward_rotation: float = 1.0
    reward_fingertip: float = 0.5
    reward_action_penalty: float = 0.02
    reward_action_rate_penalty: float = 0.01

    # Goal
    goal_rotation_axis: tuple = (0.0, 0.0, 1.0)  # rotate around z-axis
    goal_rotation_speed: float = 1.0  # rad/s


@dataclass
class PPOConfig:
    """PPO algorithm configuration."""

    # Learning
    learning_rate: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_epsilon: float = 0.2
    entropy_coef: float = 0.01
    value_coef: float = 0.5
    max_grad_norm: float = 1.0

    # Training
    num_epochs: int = 5
    num_minibatches: int = 4
    total_timesteps: int = 100_000_000
    rollout_steps: int = 16

    # Network
    hidden_dims: tuple = (256, 256, 128)
    activation: str = "elu"


@dataclass
class WandbConfig:
    """Weights & Biases configuration."""

    enabled: bool = True
    project: str = "allegro-cube-ppo"
    entity: str | None = None  # Your wandb username or team
    group: str | None = None
    tags: list[str] = field(default_factory=lambda: ["ppo", "allegro", "newton"])
    mode: str = "online"  # "online", "offline", or "disabled"
    save_model: bool = True  # Save model artifacts to wandb


@dataclass
class TrainConfig:
    """Training configuration."""

    seed: int = 42
    device: str = "cuda"
    log_interval: int = 10
    save_interval: int = 100
    eval_interval: int = 50
    checkpoint_dir: str = "checkpoints"
    experiment_name: str = "allegro_cube_ppo"

    env: EnvConfig = field(default_factory=EnvConfig)
    ppo: PPOConfig = field(default_factory=PPOConfig)
    wandb: WandbConfig = field(default_factory=WandbConfig)
