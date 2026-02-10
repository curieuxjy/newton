"""Hyperparameters for Allegro Hand Cube Rotation PPO training.

Reference: IsaacGymEnvs DextrEme (allegro_hand_dextreme.py)
"""

from dataclasses import dataclass, field


@dataclass
class EnvConfig:
    """Environment configuration."""

    num_envs: int = 4096
    episode_length: int = 600  # 10 seconds at 60Hz control

    # Simulation
    fps: int = 120
    sim_substeps: int = 1
    control_decimation: int = 2  # control at 30Hz

    # Robot
    hand_stiffness: float = 40.0
    hand_damping: float = 2.0

    # Action (DextrEme style)
    action_scale: float = 0.5  # Larger for more movement
    use_relative_control: bool = True  # Delta position control

    # Cube
    cube_size: float = 0.065
    cube_mass: float = 0.1

    # Reward weights (DextrEme style)
    # Rotation reward: 1 / (rot_dist + rot_eps) * rot_reward_scale
    rot_reward_scale: float = 1.0
    rot_eps: float = 0.1

    # Distance penalty (object too far from hand)
    dist_reward_scale: float = -10.0

    # Action penalties (very small in DextrEme)
    action_penalty_scale: float = 0.0002
    action_delta_penalty_scale: float = 0.0001

    # Velocity penalty (DextrEme: -0.05 * sum((dof_vel/4)^2))
    velocity_penalty_scale: float = 0.05
    velocity_norm: float = 4.0

    # Success/Failure
    reach_goal_bonus: float = 250.0
    fall_penalty: float = -50.0
    success_tolerance: float = 0.4  # radians (~23 degrees)
    consecutive_successes: int = 50  # Hold for N steps
    fall_dist: float = 0.3  # meters

    # Goal
    goal_rotation_speed: float = 0.0  # Static goal (DextrEme style)
    randomize_goal: bool = True

    # Domain randomization (optional)
    randomize_cube_mass: bool = False
    randomize_friction: bool = False


@dataclass
class PPOConfig:
    """PPO algorithm configuration."""

    # Learning
    learning_rate: float = 5e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_epsilon: float = 0.2
    entropy_coef: float = 0.0
    value_coef: float = 4.0
    bounds_loss_coef: float = 0.0001  # Penalize actions outside [-1, 1]
    max_grad_norm: float = 1.0

    # KL-based adaptive learning rate (rl_games style)
    kl_threshold: float = 0.016
    lr_schedule: str = "adaptive"  # "adaptive" or "fixed"

    # Training
    num_epochs: int = 5
    num_minibatches: int = 4
    total_timesteps: int = 100_000_000
    rollout_steps: int = 16

    # Network (larger for complex task)
    hidden_dims: tuple = (512, 512, 256, 128)
    activation: str = "elu"

    # Normalization (DextrEme: all True)
    normalize_input: bool = True
    normalize_value: bool = True
    normalize_advantage: bool = True
    observation_clip: float = 5.0  # DextrEme: 5.0

    # Actor initialization
    init_sigma: float = 0.0  # Initial log_std (σ=1.0 for wide exploration)


@dataclass
class WandbConfig:
    """Weights & Biases configuration."""

    enabled: bool = True
    project: str = "newton-allegro-cube"
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
