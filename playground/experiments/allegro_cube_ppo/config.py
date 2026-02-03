"""Hyperparameters for Allegro Hand Cube Rotation PPO training.

Reference: IsaacLab DextrEme / Allegro Hand environments
"""

from dataclasses import dataclass, field


@dataclass
class EnvConfig:
    """Environment configuration."""

    num_envs: int = 4096
    episode_length: int = 400  # 8 seconds at 50Hz (matching IsaacLab)

    # Simulation (IsaacLab uses 1/120s timestep with decimation 4)
    fps: int = 60
    sim_substeps: int = 2
    control_decimation: int = 2  # control at 30Hz

    # Robot
    hand_stiffness: float = 40.0  # Lower for more compliant motion
    hand_damping: float = 2.0

    # Action
    action_scale: float = 0.3  # Scale from [-1,1] to joint space

    # Cube
    cube_size: float = 0.065  # 6.5cm cube
    cube_mass: float = 0.1  # 100g

    # Reward weights (matching IsaacLab style)
    reward_dist_scale: float = -10.0  # Distance to target (negative = penalty)
    reward_rot_scale: float = 1.0  # Rotation alignment
    reward_rot_eps: float = 0.1  # Rotation reward shaping epsilon
    reward_action_penalty: float = 0.0002  # Very small (IsaacLab uses 0.0002)
    reward_action_rate_penalty: float = 0.0001
    reward_success_bonus: float = 250.0  # Bonus for reaching goal
    reward_fall_penalty: float = -50.0  # Penalty if cube falls
    reward_fingertip_scale: float = 0.5  # Fingertips near cube

    # Success/Failure thresholds
    success_tolerance: float = 0.2  # Quaternion distance threshold
    fall_height: float = 0.05  # Cube fall threshold (m)

    # Goal
    goal_rotation_axis: tuple = (0.0, 0.0, 1.0)  # rotate around z-axis
    goal_rotation_speed: float = 0.5  # rad/s (slower for learning)

    # Domain randomization (optional)
    randomize_cube_mass: bool = False
    randomize_friction: bool = False


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
    rollout_steps: int = 24  # Longer rollouts for dexterous tasks

    # Network (larger for complex task)
    hidden_dims: tuple = (512, 256, 128)
    activation: str = "elu"


@dataclass
class WandbConfig:
    """Weights & Biases configuration."""

    enabled: bool = True
    project: str = "allegro-cube-ppo"
    entity: str | None = None
    group: str | None = None
    tags: list[str] = field(default_factory=lambda: ["ppo", "allegro", "newton", "dextreme"])
    mode: str = "online"
    save_model: bool = True


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
