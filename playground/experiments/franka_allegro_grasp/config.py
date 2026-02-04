"""Configuration for Franka + Allegro cube grasping environment.

Reference: DEXTRAH (NVlabs/DEXTRAH) - dextrah_kuka_allegro_env.py
"""

from dataclasses import dataclass, field


@dataclass
class EnvConfig:
    """Environment configuration."""

    num_envs: int = 256
    episode_length: int = 400  # 8 seconds at 50Hz

    # Simulation
    fps: int = 60
    sim_substeps: int = 4
    control_decimation: int = 2  # control at 30Hz

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
    use_relative_control: bool = True  # Delta position control

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
    lift_height: float = 0.2  # How high to lift the cube above table
    goal_tolerance: float = 0.05  # Distance tolerance for success

    # Depth sensor
    use_depth_sensor: bool = True
    depth_width: int = 160
    depth_height: int = 120
    depth_fov: float = 60.0  # degrees
    depth_min: float = 0.1
    depth_max: float = 2.0

    # Reward weights (DEXTRAH style)
    # Phase 1: Reach - move hand to cube
    reach_reward_scale: float = 1.0
    reach_bonus: float = 50.0
    reach_threshold: float = 0.1  # distance to consider "reached"

    # Phase 2: Grasp - close fingers around cube
    grasp_reward_scale: float = 2.0
    grasp_bonus: float = 100.0
    finger_contact_reward: float = 10.0

    # Phase 3: Lift - lift cube to goal height
    lift_reward_scale: float = 5.0
    lift_bonus: float = 250.0

    # Penalties
    action_penalty_scale: float = 0.0001
    action_delta_penalty_scale: float = 0.0001
    velocity_penalty_scale: float = 0.01
    drop_penalty: float = -100.0

    # Success/Failure
    consecutive_successes: int = 10  # Hold for N steps
    fall_height: float = 0.3  # Below this z, cube is dropped

    # Domain randomization
    randomize_cube_pos: bool = True
    randomize_cube_mass: bool = False


@dataclass
class PPOConfig:
    """PPO algorithm configuration."""

    learning_rate: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_epsilon: float = 0.2
    entropy_coef: float = 0.01
    value_coef: float = 0.5
    max_grad_norm: float = 1.0

    num_epochs: int = 5
    num_minibatches: int = 4
    total_timesteps: int = 100_000_000
    rollout_steps: int = 24

    # Network
    hidden_dims: tuple = (512, 256, 128)
    activation: str = "elu"

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
    save_interval: int = 100
    checkpoint_dir: str = "checkpoints"
    experiment_name: str = "franka_allegro_grasp"

    env: EnvConfig = field(default_factory=EnvConfig)
    ppo: PPOConfig = field(default_factory=PPOConfig)
    wandb: WandbConfig = field(default_factory=WandbConfig)
