"""Configuration for Franka + Allegro cube grasping environment.

Reference: DEXTRAH-G (NVlabs/DEXTRAH) - dextrah_kuka_allegro_env.py
"""

from dataclasses import dataclass, field


@dataclass
class EnvConfig:
    """Environment configuration."""

    num_envs: int = 256
    episode_length: int = 600  # 10 seconds at 60Hz control (DEXTRAH: 10.0s)

    # Simulation (DEXTRAH: 120Hz physics, 60Hz fabric, 15Hz policy)
    fps: int = 120  # Physics frequency
    sim_substeps: int = 1
    control_decimation: int = 2  # 60Hz fabric (120Hz / 2)
    policy_decimation: int = 4  # 15Hz policy (60Hz fabric / 4)

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
    use_relative_control: bool = True

    # FABRICS Action Space (DEXTRAH-style 11D)
    use_fabric_actions: bool = True
    fabric_ik_damping: float = 0.1
    fabric_ik_step_size: float = 0.5
    fabric_decimation: int = 2  # Fabric steps per physics step

    # Table
    table_height: float = 0.4
    table_size: tuple = (0.6, 0.8, 0.02)
    table_pos: tuple = (-0.3, -0.5)

    # Cube
    cube_size: float = 0.05
    cube_mass: float = 0.1
    cube_spawn_pos: tuple = (-0.3, -0.5, 0.45)
    cube_spawn_noise: float = 0.05

    # Goal
    lift_height: float = 0.15
    goal_tolerance: float = 0.1

    # Depth sensor (DEXTRAH-G values)
    use_depth_sensor: bool = True
    depth_width: int = 160
    depth_height: int = 120
    depth_fov: float = 48.0  # DEXTRAH-G: 48 degrees
    depth_min: float = 0.5  # DEXTRAH-G: 0.5m
    depth_max: float = 1.3  # DEXTRAH-G: 1.3m

    # Reward weights (DEXTRAH original values)
    hand_to_object_weight: float = 1.0
    hand_to_object_sharpness: float = 10.0
    hand_to_object_dist_threshold: float = 0.3

    object_to_goal_weight: float = 5.0
    object_to_goal_sharpness: float = 15.0

    lift_weight: float = 5.0
    lift_sharpness: float = 8.5
    object_height_thresh: float = 0.15

    finger_curl_reg_weight: float = -0.01

    in_success_region_weight: float = 10.0
    object_goal_tol: float = 0.1

    # Reward scaling (DEXTRAH: reward_shaper.scale_value = 0.01)
    reward_scale: float = 0.01

    # Success/Failure
    consecutive_successes: int = 10
    fall_height: float = 0.3
    min_episode_steps: int = 60

    # Workspace bounds for out-of-reach termination (DEXTRAH-style)
    workspace_margin: float = 0.15  # Margin beyond table edges
    z_height_cutoff: float = 0.2  # Below this = fell off table

    # Domain randomization
    randomize_cube_pos: bool = True
    randomize_cube_mass: bool = False


@dataclass
class TeacherPPOConfig:
    """Teacher PPO configuration (DEXTRAH-G privileged RL)."""

    # Learning (DEXTRAH teacher: lr=3e-4, critic_lr=5e-5)
    learning_rate: float = 3e-4
    critic_lr: float = 5e-5
    lr_schedule: str = "adaptive"
    gamma: float = 0.998  # DEXTRAH: 0.998
    gae_lambda: float = 0.95

    # Clipping
    clip_epsilon: float = 0.2
    kl_threshold: float = 0.013  # DEXTRAH: 0.013
    clip_value: bool = True

    # Loss coefficients (DEXTRAH teacher values)
    entropy_coef: float = 0.002  # DEXTRAH teacher: 0.002
    value_coef: float = 4.0
    bounds_loss_coef: float = 0.005  # DEXTRAH teacher: 0.005
    max_grad_norm: float = 1.0

    # Training dynamics
    num_epochs: int = 4  # DEXTRAH: mini_epochs=4
    minibatch_size: int = 1024  # 4 minibatches with 256 envs (DEXTRAH: 16384 with 4096 envs → 4 mb)
    rollout_steps: int = 16  # DEXTRAH: horizon_length=16
    max_iterations: int = 20000  # DEXTRAH: max_epochs=20000

    # Normalization
    normalize_input: bool = True
    normalize_value: bool = True
    normalize_advantage: bool = True
    observation_clip: float = 5.0
    action_clip: float = 1.0

    # Actor network: LSTM(1024) -> MLP[512,512]
    actor_lstm_units: int = 1024
    actor_mlp_dims: tuple = (512, 512)

    # Critic network (central value): LSTM(2048) -> MLP[1024,512]
    critic_lstm_units: int = 2048
    critic_mlp_dims: tuple = (1024, 512)

    # LSTM common
    lstm_layers: int = 1
    lstm_layer_norm: bool = True
    seq_len: int = 16  # DEXTRAH: sequence_length=16

    # Activation
    activation: str = "elu"

    # Sigma
    fixed_sigma: bool = True
    init_sigma: float = 0.0


@dataclass
class StudentConfig:
    """Student distillation configuration (DEXTRAH-G depth policy)."""

    # Learning
    learning_rate: float = 1e-4
    lr_schedule: str = "adaptive"
    kl_threshold: float = 0.016
    gamma: float = 0.998
    gae_lambda: float = 0.95

    # PPO params
    clip_epsilon: float = 0.2
    entropy_coef: float = 0.0
    value_coef: float = 4.0
    bounds_loss_coef: float = 0.0001
    max_grad_norm: float = 1.0
    num_epochs: int = 4
    minibatch_size: int = 16384
    rollout_steps: int = 16

    # CNN depth encoder: [16,32,64,128] channels + LayerNorm
    cnn_channels: tuple = (16, 32, 64, 128)
    cnn_kernel_sizes: tuple = (6, 4, 4, 4)
    cnn_strides: tuple = (2, 2, 2, 2)
    cnn_output_dim: int = 128
    cnn_use_layer_norm: bool = True

    # Student LSTM
    lstm_units: int = 512
    lstm_layers: int = 1
    lstm_layer_norm: bool = True
    seq_len: int = 20  # DEXTRAH student: sequence_length=20

    # Student MLP: [512,512,256]
    mlp_dims: tuple = (512, 512, 256)

    # Auxiliary head: MLP[512,256] -> object_pos(3)
    aux_mlp_dims: tuple = (512, 256)
    aux_output_dim: int = 3  # Object position prediction

    activation: str = "elu"
    fixed_sigma: bool = True
    init_sigma: float = 0.0

    # Normalization
    normalize_input: bool = True
    normalize_value: bool = True
    normalize_advantage: bool = True
    observation_clip: float = 5.0
    action_clip: float = 1.0


@dataclass
class DistillConfig:
    """Distillation training configuration."""

    learning_rate: float = 1e-4
    warmup_steps: int = 1000
    max_iterations: int = 100000

    # Auxiliary loss weight schedule
    beta_initial: float = 1.0
    beta_final: float = 0.0
    beta_decay_iteration: int = 15000

    # Distillation loss
    use_inverse_variance_weighting: bool = True  # w = 1/sigma_teacher^2

    # Logging
    log_interval: int = 100
    save_interval: int = 5000


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
    save_interval: int = 100
    checkpoint_dir: str = "checkpoints"
    experiment_name: str = "franka_allegro_grasp"

    env: EnvConfig = field(default_factory=EnvConfig)
    teacher: TeacherPPOConfig = field(default_factory=TeacherPPOConfig)
    student: StudentConfig = field(default_factory=StudentConfig)
    distill: DistillConfig = field(default_factory=DistillConfig)
    wandb: WandbConfig = field(default_factory=WandbConfig)


# Backward compatibility alias
PPOConfig = TeacherPPOConfig
