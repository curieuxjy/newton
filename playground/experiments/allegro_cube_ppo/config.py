"""Hyperparameters for Allegro Hand Cube Rotation PPO training.

Reference: IsaacGymEnvs DextrEme (allegro_hand_dextreme.py)
Config: AllegroHandDextremeManualDR.yaml + AllegroHandDextremeManualDRPPO.yaml
Paper: "DextrEme: Transfer of Agile In-Hand Manipulation from Simulation to Reality"
"""

from dataclasses import dataclass, field


@dataclass
class EnvConfig:
    """Environment configuration (DextrEme ManualDR aligned).

    Timing:
        DextrEme uses dt=1/60, substeps=2, controlFrequencyInv=2.
        This gives physics at 120Hz effective and control at 30Hz.
        Episode: resetTime=8s at 30Hz control = 240 steps.
    """

    num_envs: int = 4096

    # Simulation timing (DextrEme: physics 120Hz, control 30Hz)
    fps: int = 60
    sim_substeps: int = 2  # 120Hz effective physics
    control_decimation: int = 2  # 30Hz control

    # Episode (DextrEme: resetTime=8s at 30Hz = 240 steps)
    episode_length: int = 240

    # Robot (Newton simulator-specific tuning)
    # Note: DextrEme original uses stiffness=2, damping=0.1 for PhysX.
    # Newton MuJoCo solver needs higher values for stable grasping.
    hand_stiffness: float = 40.0
    hand_damping: float = 2.0

    # Action control (DextrEme ManualDR: absolute position control)
    use_relative_control: bool = False
    dof_speed_scale: float = 20.0  # Only used in relative control mode
    action_scale: float = 1.0  # Only used in relative control mode

    # Action moving average EMA (DextrEme paper Section 4)
    # Paper: "annealed from 0.2 to 0.15"
    act_moving_average_upper: float = 0.2
    act_moving_average_lower: float = 0.15
    act_moving_average_schedule_steps: int = 1_000_000
    act_moving_average_schedule_freq: int = 500

    # Cube
    cube_size: float = 0.065
    cube_mass: float = 0.1

    # Reward (DextrEme: signs already included in scale values)
    # reward = dist_rew + rot_rew + action_penalty + action_delta_penalty
    #        + velocity_penalty + reach_goal_rew + fall_rew + timeout_rew
    dist_reward_scale: float = -10.0
    rot_reward_scale: float = 1.0
    rot_eps: float = 0.1
    action_penalty_scale: float = -0.0001  # Negative sign included
    action_delta_penalty_scale: float = -0.01  # Negative sign included
    # Velocity penalty is hardcoded in compute_hand_reward:
    #   coef=-0.05, max_velocity=5.0, vel_tolerance=1.0, denominator=4.0

    # Success/Failure (DextrEme ManualDR values)
    success_tolerance: float = 0.4  # radians (~23 degrees)
    max_consecutive_successes: int = 50
    reach_goal_bonus: float = 250.0
    fall_penalty: float = 0.0  # DextrEme: disabled
    fall_dist: float = 0.24
    num_success_hold_steps: int = 1  # Steps within tolerance to count as goal reached
    av_factor: float = 0.1  # EMA factor for consecutive_successes logging

    # Goal
    randomize_goal: bool = True  # apply_random_quat = True

    # Reset noise (DextrEme ManualDR)
    reset_dof_pos_noise: float = 0.5
    reset_dof_vel_noise: float = 0.0
    reset_position_noise: float = 0.03
    reset_position_noise_z: float = 0.01


@dataclass
class DRConfig:
    """Domain Randomisation / Vectorised ADR configuration.

    Modes:
        "off":    No domain randomisation
        "static": Fixed DR ranges, sample at each reset (no adaptation)
        "adr":    Full Vectorised ADR with adaptive boundary expansion

    Reference: DextrEme paper Section 4, "Vectorised Automatic Domain Randomisation"
    """

    mode: str = "adr"  # "off", "static", or "adr"

    # ADR hyperparameters (only used when mode="adr")
    adr_boundary_fraction: float = 0.4  # 40% boundary workers (evaluation)
    adr_queue_length: int = 256  # Performance queue capacity N
    adr_threshold_high: float = 20.0  # Widen boundary when mean > this
    adr_threshold_low: float = 5.0  # Tighten boundary when mean < this
    adr_clear_other_queues: bool = True  # Clear all queues on any boundary change

    # Static DR ranges (used when mode="static")
    # When mode="adr", these are ignored; ADR starts from nominal values
    hand_stiffness_range: tuple = (20.0, 80.0)
    hand_damping_range: tuple = (0.5, 5.0)
    cube_mass_range: tuple = (0.05, 0.3)
    hand_friction_range: tuple = (0.5, 2.5)
    cube_friction_range: tuple = (0.5, 2.5)
    obs_noise_range: tuple = (0.0, 0.2)
    action_noise_range: tuple = (0.0, 0.1)


@dataclass
class PPOConfig:
    """PPO algorithm configuration (DextrEme paper Section 4 aligned).

    Paper architecture:
        Actor:  LSTM(1024, layer_norm) + MLP [512, 512] + ELU
        Critic: LSTM(2048, layer_norm) + MLP [1024, 512] + ELU
        Separate LR: actor lr=1e-4 (linear schedule), critic lr=5e-5 (fixed)

    MLP fallback (simplified, no LSTM):
        Actor:  MLP [512, 512, 256, 128] + LayerNorm + ELU
        Critic: MLP [512, 512, 256, 128] + LayerNorm + ELU (separate network)
    """

    # Network type: "lstm" (paper) or "mlp" (simplified)
    network_type: str = "lstm"

    # LSTM config (paper Section 4: actor 1024, critic 2048)
    actor_lstm_size: int = 1024
    critic_lstm_size: int = 2048
    actor_mlp_dims: tuple = (512, 512)  # Post-LSTM MLP for actor
    critic_mlp_dims: tuple = (1024, 512)  # Post-LSTM MLP for critic

    # MLP-only config (fallback when network_type="mlp")
    mlp_hidden_dims: tuple = (512, 512, 256, 128)

    # Layer normalization (paper: "with layer normalization")
    use_layer_norm: bool = True
    activation: str = "elu"

    # Learning rates (paper: separate for actor and critic)
    # "best result was obtained using linear scheduling of the learning rate
    #  for the policy (start value lr=1e-4) and a fixed learning rate
    #  for the value function (lr=5e-5)"
    learning_rate: float = 1e-4  # Actor LR (start value)
    critic_learning_rate: float = 5e-5  # Critic LR (fixed)

    # LR schedule (paper: "best result was obtained using linear scheduling")
    lr_schedule: str = "linear"  # "linear" (paper best) or "adaptive" (DextrEme yaml)
    kl_threshold: float = 0.016  # Only used for adaptive schedule

    # PPO hyperparameters
    gamma: float = 0.998
    gae_lambda: float = 0.95
    clip_epsilon: float = 0.2
    entropy_coef: float = 0.0
    value_coef: float = 4.0
    bounds_loss_coef: float = 0.0001
    max_grad_norm: float = 1.0

    # Training
    num_epochs: int = 4
    minibatch_size: int = 16384  # LSTM: envs_per_batch * rollout_steps
    total_timesteps: int = 100_000_000
    rollout_steps: int = 16  # BPTT truncation length

    # Normalization (DextrEme: all True)
    normalize_input: bool = True
    normalize_value: bool = True
    normalize_advantage: bool = True
    observation_clip: float = 5.0

    # Actor initialization (rl_games: sigma_init val=0 -> log_std=0, sigma=1.0)
    init_sigma: float = 0.0


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
    dr: DRConfig = field(default_factory=DRConfig)
    wandb: WandbConfig = field(default_factory=WandbConfig)
