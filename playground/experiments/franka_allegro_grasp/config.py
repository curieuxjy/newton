"""Configuration for Franka + Allegro cube grasping environment.

Reference: DEXTRAH (NVlabs/DEXTRAH) - dextrah_kuka_allegro_env.py
"""

from dataclasses import dataclass, field


@dataclass
class EnvConfig:
    """Environment configuration."""

    num_envs: int = 256
    episode_length: int = 600  # 10 seconds at 60Hz control (DEXTRAH: 10.0s)

    # Simulation (DEXTRAH: 120Hz physics, 60Hz control)
    fps: int = 120  # Physics frequency
    sim_substeps: int = 1  # No additional substeps (already 120Hz)
    control_decimation: int = 2  # 60Hz control (120Hz / 2)

    # Robot - Franka arm
    franka_stiffness: float = 500.0
    franka_damping: float = 100.0
    franka_effort_limit: float = 80.0
    franka_armature: float = 0.1

    # Robot - Allegro hand
    hand_stiffness: float = 200.0
    hand_damping: float = 20.0
    hand_effort_limit: float = 20.0
    hand_armature: float = 0.05

    # Action
    action_scale: float = 0.1
    use_relative_control: bool = True  # Delta position control (for direct joint control)

    # FABRICS Action Space (DEXTRAH-style)
    # Action space is 11D instead of 23D:
    # - 6D palm pose (XYZ + RPY) for arm control
    # - 5D hand PCA for finger control
    # Set to False for direct 23D joint control
    use_fabric_actions: bool = True  # Default: FABRICS action space (DEXTRAH-style)

    # Fabric IK parameters
    fabric_ik_damping: float = 0.1  # Damping for differential IK
    fabric_ik_step_size: float = 0.5  # Step size for IK updates
    fabric_decimation: int = 2  # Number of fabric steps per physics step (DEXTRAH: 2)

    # Table - positioned where robot arm points (EE at approx -0.4, -0.7)
    table_height: float = 0.4
    table_size: tuple = (0.6, 0.8, 0.02)  # (x, y, z) - width, depth, thickness
    table_pos: tuple = (-0.3, -0.5)  # (x, y) center position - in front of robot arm

    # Cube
    cube_size: float = 0.05
    cube_mass: float = 0.1
    cube_spawn_pos: tuple = (-0.3, -0.5, 0.45)  # (x, y, z) - on table, reachable by robot
    cube_spawn_noise: float = 0.05  # XY randomization range

    # Goal
    lift_height: float = 0.15  # How high to lift the cube above table (DEXTRAH: object_height_thresh)
    goal_tolerance: float = 0.1  # Distance tolerance for success (DEXTRAH: object_goal_tol)

    # Depth sensor
    use_depth_sensor: bool = True
    depth_width: int = 160
    depth_height: int = 120
    depth_fov: float = 60.0  # degrees
    depth_min: float = 0.1
    depth_max: float = 2.0

    # Reward weights (DEXTRAH original values)
    # Hand-to-object reward: weight * exp(-sharpness * dist)
    hand_to_object_weight: float = 1.0
    hand_to_object_sharpness: float = 10.0
    hand_to_object_dist_threshold: float = 0.3  # meters

    # Object-to-goal reward: weight * exp(-sharpness * dist)
    object_to_goal_weight: float = 5.0
    object_to_goal_sharpness: float = 15.0  # ADR range: 15-20

    # Lift reward: weight * exp(-sharpness * vertical_error)
    lift_weight: float = 5.0
    lift_sharpness: float = 8.5
    object_height_thresh: float = 0.15  # meters for lift criteria

    # Finger curl regularization (penalty)
    finger_curl_reg_weight: float = -0.01

    # Success bonus
    in_success_region_weight: float = 10.0
    object_goal_tol: float = 0.1  # meters

    # Note: DEXTRAH does NOT use action/velocity penalties in reward
    # Only the 4 core reward components are used

    # Success/Failure
    consecutive_successes: int = 10  # Hold for N steps
    fall_height: float = 0.3  # Below this z, cube is dropped
    min_episode_steps: int = 60  # DEXTRAH minimum episode length

    # Domain randomization
    randomize_cube_pos: bool = True
    randomize_cube_mass: bool = False


@dataclass
class PPOConfig:
    """PPO algorithm configuration (DEXTRAH original values)."""

    # Learning (DEXTRAH: 5e-4 with adaptive schedule)
    learning_rate: float = 5e-4
    lr_schedule: str = "adaptive"  # "adaptive" or "linear" or "constant"
    gamma: float = 0.99
    gae_lambda: float = 0.95  # DEXTRAH: tau = 0.95

    # Clipping (DEXTRAH values)
    clip_epsilon: float = 0.2  # DEXTRAH: e_clip = 0.2
    kl_threshold: float = 0.016  # DEXTRAH: kl_threshold = 0.016
    clip_value: bool = True

    # Loss coefficients (DEXTRAH values)
    entropy_coef: float = 0.0  # DEXTRAH: 0.0
    value_coef: float = 4.0  # DEXTRAH: critic_coef = 4
    bounds_loss_coef: float = 0.0001  # DEXTRAH: bounds_loss_coef = 0.0001
    max_grad_norm: float = 1.0

    # Training dynamics (DEXTRAH values)
    num_epochs: int = 5  # DEXTRAH: mini_epochs = 5
    num_minibatches: int = 4  # Will adjust based on num_envs
    minibatch_size: int = 8192  # DEXTRAH: 8192
    total_timesteps: int = 100_000_000
    rollout_steps: int = 16  # DEXTRAH: horizon_length = 16
    max_iterations: int = 5000  # DEXTRAH: max_epochs = 5000

    # Normalization (DEXTRAH values)
    normalize_input: bool = True
    normalize_value: bool = True
    normalize_advantage: bool = True
    observation_clip: float = 5.0  # DEXTRAH: 5.0
    action_clip: float = 1.0  # DEXTRAH: 1.0

    # Network (DEXTRAH: [512, 512, 256, 128])
    hidden_dims: tuple = (512, 512, 256, 128)
    activation: str = "elu"

    # Sigma (DEXTRAH: fixed sigma)
    fixed_sigma: bool = True
    init_sigma: float = 0.0  # DEXTRAH: 0.0

    # Depth encoder (if using depth)
    depth_encoder_dims: tuple = (32, 64, 128)
    depth_latent_dim: int = 64


@dataclass
class WandbConfig:
    """Weights & Biases configuration."""

    enabled: bool = True
    project: str = "newton-franka-allegro-grasp"
    entity: str | None = None
    group: str | None = None
    tags: list[str] = field(default_factory=lambda: ["ppo", "franka", "allegro", "grasp", "newton"])
    mode: str = "online"


@dataclass
class TrainConfig:
    """Training configuration."""

    seed: int = 42
    device: str = "cuda"
    log_interval: int = 10
    save_interval: int = 5000
    checkpoint_dir: str = "checkpoints"
    experiment_name: str = "franka_allegro_grasp"

    env: EnvConfig = field(default_factory=EnvConfig)
    ppo: PPOConfig = field(default_factory=PPOConfig)
    wandb: WandbConfig = field(default_factory=WandbConfig)
