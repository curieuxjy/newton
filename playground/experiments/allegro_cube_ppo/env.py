"""Allegro Hand Cube Rotation Environment for RL training with Newton.

Reference: IsaacGymEnvs DextrEme (allegro_hand_dextreme.py)

Observation space (72 dims):
    DextrEme ManualDR actor inputs (50 dims):
        dof_pos (16), object_pose (7), goal_pose (7),
        goal_relative_rot (4), last_actions (16)
    Added for MLP (compensates for no LSTM):
        dof_vel (16), object_vels (6)

Action space (16 dims):
    Absolute position control with EMA smoothing (DextrEme ManualDR style).
    Policy outputs in [-1, 1], scaled to [joint_lower, joint_upper].
"""

from typing import Any

import numpy as np
import torch
import warp as wp

import newton
from newton import ActuatorMode

from .adr import ADRManager, make_default_adr_params
from .config import DRConfig, EnvConfig


def _scale(x: torch.Tensor, lower: torch.Tensor, upper: torch.Tensor) -> torch.Tensor:
    """Map [-1, 1] -> [lower, upper] (DextrEme scale function)."""
    return (0.5 * (x + 1.0)) * (upper - lower) + lower


def _unscale(x: torch.Tensor, lower: torch.Tensor, upper: torch.Tensor) -> torch.Tensor:
    """Map [lower, upper] -> [-1, 1] (DextrEme unscale function)."""
    return (2.0 * x - upper - lower) / (upper - lower)


def _quat_mul(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
    """Multiply two quaternions (x, y, z, w format)."""
    x1, y1, z1, w1 = q1[..., 0], q1[..., 1], q1[..., 2], q1[..., 3]
    x2, y2, z2, w2 = q2[..., 0], q2[..., 1], q2[..., 2], q2[..., 3]

    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2

    return torch.stack([x, y, z, w], dim=-1)


def _quat_conjugate(q: torch.Tensor) -> torch.Tensor:
    """Compute quaternion conjugate (x, y, z, w format)."""
    return torch.stack([-q[..., 0], -q[..., 1], -q[..., 2], q[..., 3]], dim=-1)


def _random_quaternion(n: int, device: str) -> torch.Tensor:
    """Generate random unit quaternions (uniform on SO(3), x,y,z,w format).

    Uses the Shoemake method (matches DextrEme's get_random_quat).
    """
    uvw = torch.rand(n, 3, device=device)
    q = torch.zeros(n, 4, device=device)

    sqrt1_u0 = torch.sqrt(1 - uvw[:, 0])
    sqrt_u0 = torch.sqrt(uvw[:, 0])

    # DextrEme ordering: q_w, q_x, q_y, q_z then stacked as (x,y,z,w)
    q_w = sqrt1_u0 * torch.sin(2 * np.pi * uvw[:, 1])
    q_x = sqrt1_u0 * torch.cos(2 * np.pi * uvw[:, 1])
    q_y = sqrt_u0 * torch.sin(2 * np.pi * uvw[:, 2])
    q_z = sqrt_u0 * torch.cos(2 * np.pi * uvw[:, 2])

    q[:, 0] = q_x  # x
    q[:, 1] = q_y  # y
    q[:, 2] = q_z  # z
    q[:, 3] = q_w  # w

    return q


class AllegroHandCubeEnv:
    """Vectorized environment for Allegro Hand cube rotation (DextrEme aligned).

    Key alignment with DextrEme ManualDR:
    - Absolute position control with EMA smoothing
    - 30Hz control frequency (physics 120Hz)
    - Exact reward function from compute_hand_reward
    - hold_count-based success tracking (50 goals per episode)
    - Progress buffer reset when near goal

    Known deviations:
    - MLP network instead of LSTM + MLP
    - Velocity observations added (compensates for no LSTM)
    - World-frame poses instead of wrist-relative
    - Newton MuJoCo solver instead of IsaacGym PhysX
    """

    def __init__(
        self,
        config: EnvConfig,
        device: str = "cuda",
        headless: bool = True,
        dr_config: DRConfig | None = None,
    ):
        self.config = config
        self.num_envs = config.num_envs
        self.device = wp.get_device(device)
        self.torch_device = "cuda" if self.device.is_cuda else "cpu"
        self.headless = headless
        self.dr_config = dr_config if dr_config is not None else DRConfig(mode="off")

        # Timing (DextrEme: dt=1/60, substeps=2, controlFreqInv=2)
        self.fps = config.fps
        self.frame_dt = 1.0 / self.fps
        self.sim_substeps = config.sim_substeps
        self.sim_dt = self.frame_dt / self.sim_substeps
        self.control_decimation = config.control_decimation

        # Episode
        self.max_episode_length = config.episode_length

        # Build simulation
        self._build_simulation()

        # Observation and action spaces
        # 72 = 16(dof_pos) + 16(dof_vel) + 7(obj_pose) + 6(obj_vels)
        #     + 7(goal_pose) + 4(goal_rel_rot) + 16(last_actions)
        self.num_obs = 72
        self.num_actions = 16

        # RL buffers (DextrEme style: integer reset_buf, not bool)
        self.obs_buf = torch.zeros(
            self.num_envs, self.num_obs, dtype=torch.float32, device=self.torch_device
        )
        self.reward_buf = torch.zeros(self.num_envs, dtype=torch.float32, device=self.torch_device)
        self.reset_buf = torch.ones(self.num_envs, dtype=torch.long, device=self.torch_device)
        self.reset_goal_buf = torch.zeros(self.num_envs, dtype=torch.long, device=self.torch_device)
        self.progress_buf = torch.zeros(self.num_envs, dtype=torch.long, device=self.torch_device)
        self.info_buf: dict[str, Any] = {}

        # Action buffers
        self.actions = torch.zeros(
            self.num_envs, self.num_actions, dtype=torch.float32, device=self.torch_device
        )
        self.prev_actions = torch.zeros_like(self.actions)

        # Joint target buffers (DextrEme: cur_targets, prev_targets)
        self.cur_targets = torch.zeros(
            self.num_envs, self.num_actions, dtype=torch.float32, device=self.torch_device
        )
        self.prev_targets = torch.zeros_like(self.cur_targets)

        # Action moving average (DextrEme ManualDR)
        self.act_moving_average = config.act_moving_average_upper

        # Goal state (DextrEme: goal_pos fixed, goal_rot random)
        self.goal_pos = torch.zeros(
            self.num_envs, 3, dtype=torch.float32, device=self.torch_device
        )
        self.goal_rot = torch.zeros(
            self.num_envs, 4, dtype=torch.float32, device=self.torch_device
        )
        self.goal_rot[:, 3] = 1.0  # identity quaternion

        # Success tracking (DextrEme style)
        self.successes = torch.zeros(self.num_envs, dtype=torch.long, device=self.torch_device)
        self.hold_count_buf = torch.zeros(
            self.num_envs, dtype=torch.long, device=self.torch_device
        )
        self.consecutive_successes = torch.zeros(
            1, dtype=torch.float32, device=self.torch_device
        )

        # Frame counter (matches gym.get_frame_count)
        self.frame_count = 0

        # Reward components for logging
        self.reward_components: dict[str, float] = {}

        # --- Domain Randomisation / ADR ---
        self._init_domain_randomisation()

        # Initial state storage
        self._store_initial_state()

        # CUDA graph (disabled)
        self.graph = None

    def _build_simulation(self):
        """Build the Newton simulation with Allegro hand and cube."""
        hand_builder = newton.ModelBuilder()
        newton.solvers.SolverMuJoCo.register_custom_attributes(hand_builder)

        # Contact parameters for good grasping
        hand_builder.default_shape_cfg.ke = 1.0e4
        hand_builder.default_shape_cfg.kd = 1.0e3
        hand_builder.default_shape_cfg.mu = 1.2

        # Load Allegro hand from downloaded asset
        asset_path = newton.utils.download_asset("wonik_allegro")
        asset_file = str(asset_path / "usd" / "allegro_left_hand_with_cube.usda")

        hand_builder.add_usd(
            asset_file,
            xform=wp.transform(wp.vec3(0, 0, 0.3)),
            ignore_paths=[".*Dummy", ".*CollisionPlane", ".*goal"],
        )

        self.num_hand_dofs = hand_builder.joint_dof_count
        self.num_hand_q = hand_builder.joint_count
        self.num_hand_bodies = hand_builder.body_count
        self.num_shapes_per_env = hand_builder.shape_count

        # Set joint control parameters
        for i in range(self.num_hand_dofs):
            hand_builder.joint_target_ke[i] = self.config.hand_stiffness
            hand_builder.joint_target_kd[i] = self.config.hand_damping
            hand_builder.joint_target_pos[i] = 0.0
            hand_builder.joint_act_mode[i] = int(ActuatorMode.POSITION)

        # Actuated revolute joints (indices into joint_q)
        actuated_joint_indices = [2, 3, 4, 5, 6, 7, 8, 9, 11, 12, 13, 14, 16, 17, 18, 19]
        self.joint_lower = torch.tensor(
            [hand_builder.joint_limit_lower[i] for i in actuated_joint_indices],
            dtype=torch.float32,
            device=self.torch_device,
        )
        self.joint_upper = torch.tensor(
            [hand_builder.joint_limit_upper[i] for i in actuated_joint_indices],
            dtype=torch.float32,
            device=self.torch_device,
        )
        # Clamp extreme values (thumb joints have unlimited range in USD)
        self.joint_lower = torch.clamp(self.joint_lower, min=-3.14)
        self.joint_upper = torch.clamp(self.joint_upper, max=3.14)

        # Replicate for all environments
        builder = newton.ModelBuilder()
        builder.replicate(hand_builder, self.num_envs, spacing=(0.5, 0.5, 0.0))

        builder.default_shape_cfg.ke = 1.0e4
        builder.default_shape_cfg.kd = 1.0e3
        builder.add_ground_plane()

        self.model = builder.finalize()

        # Solver (contact buffers scale with num_envs)
        # Base: ~75 contacts per 1024 envs for njmax, ~50 for nconmax
        njmax = max(300, (self.num_envs * 75 + 1023) // 1024)
        nconmax = max(200, (self.num_envs * 50 + 1023) // 1024)
        self.solver = newton.solvers.SolverMuJoCo(
            self.model,
            solver="newton",
            integrator="implicitfast",
            njmax=njmax,
            nconmax=nconmax,
            impratio=10.0,
            cone="elliptic",
            iterations=100,
            ls_iterations=50,
            use_mujoco_cpu=False,
        )
        print(f"[INFO] MuJoCo solver: njmax={njmax}, nconmax={nconmax}")

        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()
        self.contacts = self.model.collide(self.state_0)

        # Per-environment layout
        self.joint_q_per_env = self.state_0.joint_q.shape[0] // self.num_envs
        self.joint_qd_per_env = self.state_0.joint_qd.shape[0] // self.num_envs

        self.cube_body_offset = self.num_hand_bodies - 1
        self.hand_joint_offset = 0

        # Cube free joint offsets within per-env joint_q/qd
        # The cube free joint is the last joint: 7 elements in joint_q, 6 in joint_qd
        self.cube_q_offset = self.joint_q_per_env - 7
        self.cube_qd_offset = self.joint_qd_per_env - 6

    def _store_initial_state(self):
        """Store initial state for resetting environments."""
        self.initial_joint_q = wp.clone(self.state_0.joint_q)
        self.initial_joint_qd = wp.clone(self.state_0.joint_qd)
        newton.eval_fk(self.model, self.state_0.joint_q, self.state_0.joint_qd, self.state_0)

        # Store initial cube position per env (for goal_pos)
        body_q_np = self.state_0.body_q.numpy().reshape(self.num_envs, self.num_hand_bodies, 7)
        self.initial_cube_pos = torch.from_numpy(
            body_q_np[:, self.cube_body_offset, :3].copy()
        ).to(self.torch_device)

        # Store initial hand base position per env
        self.hand_base_pos = torch.from_numpy(body_q_np[:, 0, :3].copy()).to(self.torch_device)

        # Initialize targets to initial joint positions
        self.cur_targets = torch.zeros(
            self.num_envs, self.num_actions, dtype=torch.float32, device=self.torch_device
        )
        self.prev_targets = self.cur_targets.clone()

    def _init_domain_randomisation(self):
        """Initialise DR buffers and optional ADR manager."""
        self.adr: ADRManager | None = None

        # Per-env noise scales (updated by DR/ADR on each reset)
        self.obs_noise_scale = torch.zeros(self.num_envs, device=self.torch_device)
        self.action_noise_scale = torch.zeros(self.num_envs, device=self.torch_device)

        dr = self.dr_config
        if dr.mode == "off":
            return

        if dr.mode == "adr":
            params = make_default_adr_params(
                hand_stiffness=self.config.hand_stiffness,
                hand_damping=self.config.hand_damping,
                cube_mass=self.config.cube_mass,
            )
            self.adr = ADRManager(
                num_envs=self.num_envs,
                device=self.torch_device,
                params=params,
                boundary_fraction=dr.adr_boundary_fraction,
                queue_length=dr.adr_queue_length,
                threshold_high=dr.adr_threshold_high,
                threshold_low=dr.adr_threshold_low,
                clear_other_queues=dr.adr_clear_other_queues,
            )
            print(f"[INFO] ADR enabled: {len(params)} params, "
                  f"{dr.adr_boundary_fraction*100:.0f}% boundary workers")

        elif dr.mode == "static":
            print(f"[INFO] Static DR enabled")

    def _apply_domain_randomization(self, env_ids: torch.Tensor):
        """Sample and apply domain randomisation for reset environments.

        For ADR mode: samples come from ADRManager (boundary/rollout aware).
        For static mode: samples are uniform from config ranges.
        """
        if self.dr_config.mode == "off":
            return

        num = len(env_ids)
        if num == 0:
            return

        dr = self.dr_config

        # --- Sample DR values ---
        if self.dr_config.mode == "adr" and self.adr is not None:
            dr_vals = self.adr.sample_all(env_ids)
        else:
            # Static DR: uniform sample from config ranges
            dr_vals = {}
            lo, hi = dr.hand_stiffness_range
            dr_vals["hand_stiffness"] = torch.rand(num, device=self.torch_device) * (hi - lo) + lo
            lo, hi = dr.hand_damping_range
            dr_vals["hand_damping"] = torch.rand(num, device=self.torch_device) * (hi - lo) + lo
            lo, hi = dr.cube_mass_range
            dr_vals["cube_mass"] = torch.rand(num, device=self.torch_device) * (hi - lo) + lo
            lo, hi = dr.hand_friction_range
            dr_vals["hand_friction"] = torch.rand(num, device=self.torch_device) * (hi - lo) + lo
            lo, hi = dr.cube_friction_range
            dr_vals["cube_friction"] = torch.rand(num, device=self.torch_device) * (hi - lo) + lo
            lo, hi = dr.obs_noise_range
            dr_vals["obs_noise"] = torch.rand(num, device=self.torch_device) * (hi - lo) + lo
            lo, hi = dr.action_noise_range
            dr_vals["action_noise"] = torch.rand(num, device=self.torch_device) * (hi - lo) + lo

        env_ids_np = env_ids.cpu().numpy()

        # --- Apply joint stiffness/damping via model arrays ---
        model_ke = self.model.joint_target_ke.numpy()
        model_kd = self.model.joint_target_kd.numpy()
        stiffness_np = dr_vals["hand_stiffness"].cpu().numpy()
        damping_np = dr_vals["hand_damping"].cpu().numpy()

        for i, eid in enumerate(env_ids_np):
            start = eid * self.joint_qd_per_env + self.hand_joint_offset
            end = start + self.num_actions
            model_ke[start:end] = stiffness_np[i]
            model_kd[start:end] = damping_np[i]

        self.model.joint_target_ke = wp.array(model_ke, dtype=wp.float32, device=self.device)
        self.model.joint_target_kd = wp.array(model_kd, dtype=wp.float32, device=self.device)

        # --- Apply cube mass ---
        cube_mass_np = dr_vals["cube_mass"].cpu().numpy()
        try:
            body_mass = self.model.body_mass.numpy()
            body_inv_mass = self.model.body_inv_mass.numpy()
            for i, eid in enumerate(env_ids_np):
                body_idx = eid * self.num_hand_bodies + self.cube_body_offset
                body_mass[body_idx] = cube_mass_np[i]
                body_inv_mass[body_idx] = 1.0 / max(cube_mass_np[i], 1e-6)
            self.model.body_mass = wp.array(body_mass, dtype=wp.float32, device=self.device)
            self.model.body_inv_mass = wp.array(body_inv_mass, dtype=wp.float32, device=self.device)
        except Exception:
            pass  # Model may not expose writable body_mass

        # --- Apply friction ---
        hand_friction_np = dr_vals["hand_friction"].cpu().numpy()
        cube_friction_np = dr_vals["cube_friction"].cpu().numpy()
        try:
            shape_mu = self.model.shape_material_mu.numpy()
            for i, eid in enumerate(env_ids_np):
                shape_start = eid * self.num_shapes_per_env
                # Hand shapes (all except last = cube)
                shape_mu[shape_start : shape_start + self.num_shapes_per_env - 1] = hand_friction_np[i]
                # Cube shape (last)
                shape_mu[shape_start + self.num_shapes_per_env - 1] = cube_friction_np[i]
            self.model.shape_material_mu = wp.array(shape_mu, dtype=wp.float32, device=self.device)
        except Exception:
            pass  # Model may not expose writable shape_material_mu

        # --- Store per-env noise scales ---
        self.obs_noise_scale[env_ids] = dr_vals["obs_noise"]
        self.action_noise_scale[env_ids] = dr_vals["action_noise"]

    def _simulate_step(self):
        """Run one frame of simulation."""
        self.contacts = self.model.collide(self.state_0)

        for _ in range(self.sim_substeps):
            self.state_0.clear_forces()
            self.solver.step(
                self.state_0, self.state_1, self.control, self.contacts, self.sim_dt
            )
            self.state_0, self.state_1 = self.state_1, self.state_0

    def _update_act_moving_average(self):
        """Update action moving average schedule (DextrEme style)."""
        cfg = self.config
        if self.frame_count > 0 and self.frame_count % cfg.act_moving_average_schedule_freq == 0:
            sched_scaling = (
                1.0
                / cfg.act_moving_average_schedule_steps
                * min(self.frame_count, cfg.act_moving_average_schedule_steps)
            )
            self.act_moving_average = cfg.act_moving_average_upper + (
                cfg.act_moving_average_lower - cfg.act_moving_average_upper
            ) * sched_scaling

    def reset(self, env_ids: torch.Tensor | None = None) -> torch.Tensor:
        """Reset specified environments (DextrEme style)."""
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.torch_device)

        num_reset = len(env_ids)
        if num_reset == 0:
            return self._compute_observations()

        env_ids_np = env_ids.cpu().numpy()

        # --- Reset joint states (DextrEme style) ---
        joint_q_np = self.state_0.joint_q.numpy().reshape(self.num_envs, self.joint_q_per_env)
        joint_qd_np = self.state_0.joint_qd.numpy().reshape(self.num_envs, self.joint_qd_per_env)
        initial_q_np = self.initial_joint_q.numpy().reshape(self.num_envs, self.joint_q_per_env)
        initial_qd_np = self.initial_joint_qd.numpy().reshape(
            self.num_envs, self.joint_qd_per_env
        )

        # Reset all joint_q/qd to initial values first
        joint_q_np[env_ids_np] = initial_q_np[env_ids_np]
        joint_qd_np[env_ids_np] = initial_qd_np[env_ids_np]

        # Randomize hand joint positions (DextrEme reset_idx style)
        # pos = default_pos + reset_dof_pos_noise * rand_delta
        # where rand_delta is within [joint_lower, joint_upper] range
        rand_floats = np.random.uniform(-1.0, 1.0, size=(num_reset, self.num_actions)).astype(
            np.float32
        )
        lower_np = self.joint_lower.cpu().numpy()
        upper_np = self.joint_upper.cpu().numpy()
        default_pos = np.zeros(self.num_actions, dtype=np.float32)  # DextrEme: default = 0

        delta_max = upper_np - default_pos
        delta_min = lower_np - default_pos
        rand_01 = (rand_floats + 1.0) / 2.0  # [0, 1]
        rand_delta = delta_min + (delta_max - delta_min) * rand_01
        pos = default_pos + self.config.reset_dof_pos_noise * rand_delta

        # Write actuated joints into joint_q
        actuated_indices = [2, 3, 4, 5, 6, 7, 8, 9, 11, 12, 13, 14, 16, 17, 18, 19]
        for j_idx, q_idx in enumerate(actuated_indices):
            joint_q_np[env_ids_np, q_idx] = pos[:, j_idx]

        # Reset velocities to 0
        joint_qd_np[env_ids_np] = 0.0

        # Randomize cube position (DextrEme style)
        cube_rand = np.random.uniform(-1.0, 1.0, size=(num_reset, 3)).astype(np.float32)
        init_cube_q = initial_q_np[env_ids_np, self.cube_q_offset : self.cube_q_offset + 7]

        # Position noise
        init_cube_q[:, 0] += self.config.reset_position_noise * cube_rand[:, 0]  # x
        init_cube_q[:, 1] += self.config.reset_position_noise * cube_rand[:, 1]  # y
        init_cube_q[:, 2] += self.config.reset_position_noise_z * cube_rand[:, 2]  # z

        # Random cube rotation (DextrEme: apply_random_quat = True)
        random_rot = _random_quaternion(num_reset, self.torch_device)
        init_cube_q[:, 3:7] = random_rot.cpu().numpy()

        joint_q_np[env_ids_np, self.cube_q_offset : self.cube_q_offset + 7] = init_cube_q

        # Reset cube velocity to 0
        joint_qd_np[env_ids_np, self.cube_qd_offset : self.cube_qd_offset + 6] = 0.0

        # Write back to simulation
        self.state_0.joint_q = wp.array(joint_q_np.flatten(), dtype=wp.float32, device=self.device)
        self.state_0.joint_qd = wp.array(
            joint_qd_np.flatten(), dtype=wp.float32, device=self.device
        )

        # Recompute FK
        newton.eval_fk(self.model, self.state_0.joint_q, self.state_0.joint_qd, self.state_0)

        # --- Reset targets (DextrEme: prev_targets = cur_targets = current pos) ---
        pos_torch = torch.from_numpy(pos).to(self.torch_device)
        self.cur_targets[env_ids] = pos_torch
        self.prev_targets[env_ids] = pos_torch

        # --- Reset goal (DextrEme style) ---
        self.reset_target_pose(env_ids)

        # --- Reset tracking buffers ---
        self.progress_buf[env_ids] = 0
        self.reset_buf[env_ids] = 0
        self.reset_goal_buf[env_ids] = 0
        self.successes[env_ids] = 0
        self.hold_count_buf[env_ids] = 0

        # --- Reset action buffers ---
        self.prev_actions[env_ids] = 0.0
        self.actions[env_ids] = 0.0

        # --- Update goal_pos (initial cube position) ---
        body_q_np = self.state_0.body_q.numpy().reshape(self.num_envs, self.num_hand_bodies, 7)
        self.goal_pos[env_ids] = torch.from_numpy(
            body_q_np[env_ids_np, self.cube_body_offset, :3].copy()
        ).to(self.torch_device)
        self.hand_base_pos[env_ids] = torch.from_numpy(
            body_q_np[env_ids_np, 0, :3].copy()
        ).to(self.torch_device)

        return self._compute_observations()

    def reset_target_pose(self, env_ids: torch.Tensor):
        """Generate new random goal rotation (DextrEme style)."""
        num = len(env_ids)
        if self.config.randomize_goal:
            self.goal_rot[env_ids] = _random_quaternion(num, self.torch_device)
        else:
            # Z-axis rotation only
            angles = torch.rand(num, device=self.torch_device) * 2 * np.pi
            self.goal_rot[env_ids, 0] = 0.0
            self.goal_rot[env_ids, 1] = 0.0
            self.goal_rot[env_ids, 2] = torch.sin(angles / 2)
            self.goal_rot[env_ids, 3] = torch.cos(angles / 2)
        self.reset_goal_buf[env_ids] = 0

    def step(self, actions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict]:
        """Take a step in all environments (DextrEme step flow)."""
        # Update action moving average schedule
        self._update_act_moving_average()

        # --- Handle resets from previous step (DextrEme pre_physics_step) ---
        # Reset goals for envs that reached goal
        goal_env_ids = self.reset_goal_buf.nonzero(as_tuple=False).squeeze(-1)
        if len(goal_env_ids) > 0:
            self.reset_target_pose(goal_env_ids)

        # Reset terminated envs
        env_ids_reset = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        if len(env_ids_reset) > 0:
            # ADR: update queues with finishing envs' performance, then recycle
            if self.adr is not None:
                self.adr.update(env_ids_reset, self.successes)
                self.adr.recycle_envs(env_ids_reset)

            self.reset(env_ids_reset)

            # Apply domain randomisation to reset envs
            self._apply_domain_randomization(env_ids_reset)

        # --- Apply actions ---
        self.prev_actions = self.actions.clone()
        self.actions = torch.clamp(actions.clone(), -1.0, 1.0)

        self._apply_actions()

        # --- Step simulation ---
        for _ in range(self.control_decimation):
            self._simulate_step()

        self.frame_count += self.control_decimation

        # --- Post physics (DextrEme post_physics_step) ---
        self.progress_buf += 1

        # Compute observations
        obs = self._compute_observations()

        # Compute rewards and check termination (matches compute_hand_reward exactly)
        self._compute_rewards_and_dones()

        # Update prev_targets (DextrEme: done after reward computation)
        self.prev_targets = self.cur_targets.clone()

        return obs, self.reward_buf, self.reset_buf.bool(), self.info_buf

    def _apply_actions(self):
        """Apply actions using DextrEme-style control + EMA smoothing."""
        # Apply action noise (DR/ADR: per-env scale)
        noisy_actions = self.actions
        if self.dr_config.mode != "off" and self.action_noise_scale.any():
            act_noise = torch.randn_like(self.actions) * self.action_noise_scale.unsqueeze(1)
            noisy_actions = torch.clamp(self.actions + act_noise, -1.0, 1.0)

        if self.config.use_relative_control:
            # DextrEme relative: targets = prev + speed_scale * dt * actions
            targets = (
                self.prev_targets
                + self.config.dof_speed_scale * self.frame_dt * noisy_actions
            )
        else:
            # DextrEme absolute: scale [-1, 1] -> [joint_lower, joint_upper]
            targets = _scale(noisy_actions, self.joint_lower, self.joint_upper)

        # Action moving average EMA (DextrEme: always applied)
        self.cur_targets = (
            self.act_moving_average * targets
            + (1.0 - self.act_moving_average) * self.prev_targets
        )

        # Clamp to joint limits
        self.cur_targets = torch.clamp(self.cur_targets, self.joint_lower, self.joint_upper)

        # Apply to Newton control buffer
        self._set_joint_targets(self.cur_targets)

    def _set_joint_targets(self, target_pos: torch.Tensor):
        """Write joint position targets to Newton control buffer."""
        control_np = self.control.joint_target_pos.numpy()
        target_np = target_pos.cpu().numpy()

        env_indices = np.arange(self.num_envs)
        starts = env_indices * self.joint_qd_per_env + self.hand_joint_offset

        for j in range(self.num_actions):
            control_np[starts + j] = target_np[:, j]

        self.control.joint_target_pos = wp.array(
            control_np, dtype=wp.float32, device=self.device
        )

    def _compute_observations(self) -> torch.Tensor:
        """Compute observations (DextrEme-compatible + velocity for MLP)."""
        joint_q = torch.from_numpy(self.state_0.joint_q.numpy()).to(self.torch_device)
        joint_qd = torch.from_numpy(self.state_0.joint_qd.numpy()).to(self.torch_device)
        body_q = torch.from_numpy(self.state_0.body_q.numpy()).to(self.torch_device)
        body_qd = torch.from_numpy(self.state_0.body_qd.numpy()).to(self.torch_device)

        joint_q_r = joint_q.reshape(self.num_envs, self.joint_q_per_env)
        joint_qd_r = joint_qd.reshape(self.num_envs, self.joint_qd_per_env)
        body_q_r = body_q.reshape(self.num_envs, self.num_hand_bodies, 7)
        body_qd_r = body_qd.reshape(self.num_envs, self.num_hand_bodies, 6)

        # Extract actuated joints
        joint_pos = joint_q_r[:, self.hand_joint_offset : self.hand_joint_offset + self.num_actions]
        joint_vel = joint_qd_r[
            :, self.hand_joint_offset : self.hand_joint_offset + self.num_actions
        ]

        # Cube state
        cube_pos = body_q_r[:, self.cube_body_offset, :3]
        cube_rot = body_q_r[:, self.cube_body_offset, 3:7]
        cube_linvel = body_qd_r[:, self.cube_body_offset, :3]
        cube_angvel = body_qd_r[:, self.cube_body_offset, 3:6]
        hand_pos = body_q_r[:, 0, :3]

        # --- Build observation (72 dims) ---
        idx = 0

        # [0:16] dof_pos: normalized to [-1, 1] (DextrEme unscale)
        self.obs_buf[:, idx : idx + 16] = _unscale(joint_pos, self.joint_lower, self.joint_upper)
        idx += 16

        # [16:32] dof_vel: raw joint velocities (added for MLP, not in original actor)
        self.obs_buf[:, idx : idx + 16] = joint_vel
        idx += 16

        # [32:39] object_pose: [pos_relative(3), quat(4)]
        # Position relative to hand base (simplification of wrist-relative)
        self.obs_buf[:, idx : idx + 3] = cube_pos - hand_pos
        self.obs_buf[:, idx + 3 : idx + 7] = cube_rot
        idx += 7

        # [39:45] object_vels: [linvel(3), angvel*0.2(3)] (added for MLP)
        # DextrEme ManualDR: vel_obs_scale=0.2 for angular velocity
        self.obs_buf[:, idx : idx + 3] = cube_linvel
        self.obs_buf[:, idx + 3 : idx + 6] = 0.2 * cube_angvel
        idx += 6

        # [45:52] goal_pose: [pos_relative(3), quat(4)]
        self.obs_buf[:, idx : idx + 3] = self.goal_pos - hand_pos
        self.obs_buf[:, idx + 3 : idx + 7] = self.goal_rot
        idx += 7

        # [52:56] goal_relative_rot: quat_mul(object_rot, conjugate(goal_rot))
        self.obs_buf[:, idx : idx + 4] = _quat_mul(cube_rot, _quat_conjugate(self.goal_rot))
        idx += 4

        # [56:72] last_actions
        self.obs_buf[:, idx : idx + 16] = self.prev_actions
        idx += 16

        # Apply observation noise (DR/ADR: per-env scale)
        if self.dr_config.mode != "off":
            noise_scale = self.obs_noise_scale.unsqueeze(1)  # [N, 1]
            if noise_scale.any():
                obs_noise = torch.randn_like(self.obs_buf) * noise_scale
                self.obs_buf = self.obs_buf + obs_noise

        # Sanitize
        self.obs_buf = torch.nan_to_num(self.obs_buf, nan=0.0, posinf=5.0, neginf=-5.0)
        self.obs_buf = torch.clamp(self.obs_buf, -5.0, 5.0)

        return self.obs_buf

    def _compute_rewards_and_dones(self):
        """Compute rewards and termination (matches DextrEme compute_hand_reward exactly)."""
        cfg = self.config

        body_q = torch.from_numpy(self.state_0.body_q.numpy()).to(self.torch_device)
        joint_qd = torch.from_numpy(self.state_0.joint_qd.numpy()).to(self.torch_device)

        body_q_r = body_q.reshape(self.num_envs, self.num_hand_bodies, 7)
        joint_qd_r = joint_qd.reshape(self.num_envs, self.joint_qd_per_env)

        object_pos = body_q_r[:, self.cube_body_offset, :3]
        object_rot = body_q_r[:, self.cube_body_offset, 3:7]
        target_pos = self.goal_pos
        target_rot = self.goal_rot
        hand_dof_vel = joint_qd_r[
            :, self.hand_joint_offset : self.hand_joint_offset + self.num_actions
        ]

        # --- Distance from object to goal position ---
        goal_dist = torch.norm(object_pos - target_pos, p=2, dim=-1)

        # --- Rotation distance ---
        quat_diff = _quat_mul(object_rot, _quat_conjugate(target_rot))
        rot_dist = 2.0 * torch.asin(
            torch.clamp(torch.norm(quat_diff[:, 0:3], p=2, dim=-1), max=1.0)
        )

        # --- Reward components (DextrEme compute_hand_reward, signs included) ---
        dist_rew = goal_dist * cfg.dist_reward_scale  # -10 * dist

        rot_rew = 1.0 / (torch.abs(rot_dist) + cfg.rot_eps) * cfg.rot_reward_scale

        action_penalty = cfg.action_penalty_scale * torch.sum(self.actions**2, dim=-1)

        # Action delta: cur_targets - prev_targets (DextrEme uses target delta, not action delta)
        action_delta_penalty = cfg.action_delta_penalty_scale * torch.sum(
            (self.cur_targets - self.prev_targets) ** 2, dim=-1
        )

        # Velocity penalty (DextrEme hardcoded: coef=-0.05, denom=4.0)
        velocity_penalty_coef = -0.05
        max_velocity = 5.0
        vel_tolerance = 1.0
        velocity_penalty = velocity_penalty_coef * torch.sum(
            (hand_dof_vel / (max_velocity - vel_tolerance)) ** 2, dim=-1
        )

        # --- Success tracking (DextrEme hold_count style) ---
        goal_reached = (torch.abs(rot_dist) <= cfg.success_tolerance).long()
        self.hold_count_buf = torch.where(
            goal_reached > 0, self.hold_count_buf + 1, torch.zeros_like(self.hold_count_buf)
        )

        goal_resets = (self.hold_count_buf > cfg.num_success_hold_steps).long()
        self.successes = self.successes + goal_resets

        # Success bonus
        reach_goal_rew = goal_resets.float() * cfg.reach_goal_bonus

        # Mark goals for reset
        self.reset_goal_buf = torch.where(
            goal_resets > 0, torch.ones_like(self.reset_goal_buf), self.reset_goal_buf
        )
        self.hold_count_buf = torch.where(
            goal_resets > 0, torch.zeros_like(self.hold_count_buf), self.hold_count_buf
        )

        # Fall penalty
        fall_rew = (goal_dist >= cfg.fall_dist).float() * cfg.fall_penalty

        # --- Termination (DextrEme style) ---
        resets = torch.where(
            goal_dist >= cfg.fall_dist, torch.ones_like(self.reset_buf), torch.zeros_like(self.reset_buf)
        )

        if cfg.max_consecutive_successes > 0:
            # Reset progress when near goal (DextrEme: extends episode)
            self.progress_buf = torch.where(
                torch.abs(rot_dist) <= cfg.success_tolerance,
                torch.zeros_like(self.progress_buf),
                self.progress_buf,
            )
            # Reset when enough goals achieved
            resets = torch.where(
                self.successes >= cfg.max_consecutive_successes,
                torch.ones_like(resets),
                resets,
            )

        timed_out = self.progress_buf >= self.max_episode_length - 1
        resets = torch.where(timed_out, torch.ones_like(resets), resets)

        # Timeout penalty (0 since fall_penalty=0)
        timeout_rew = timed_out.float() * 0.5 * cfg.fall_penalty

        # --- Total reward (DextrEme: all components summed) ---
        self.reward_buf = (
            dist_rew
            + rot_rew
            + action_penalty
            + action_delta_penalty
            + velocity_penalty
            + reach_goal_rew
            + fall_rew
            + timeout_rew
        )

        self.reset_buf = resets

        # --- EMA consecutive successes for logging (DextrEme style) ---
        num_resets = torch.sum(resets)
        if num_resets > 0:
            finished_cons_successes = torch.sum(self.successes.float() * resets.float())
            self.consecutive_successes = (
                cfg.av_factor * finished_cons_successes / num_resets
                + (1.0 - cfg.av_factor) * self.consecutive_successes
            )

        # --- Store reward components for logging ---
        self.reward_components = {
            "dist_rew": dist_rew.mean().item(),
            "rot_rew": rot_rew.mean().item(),
            "action_penalty": action_penalty.mean().item(),
            "action_delta_penalty": action_delta_penalty.mean().item(),
            "velocity_penalty": velocity_penalty.mean().item(),
            "reach_goal_rew": reach_goal_rew.mean().item(),
            "fall_rew": fall_rew.mean().item(),
            "rot_dist": rot_dist.mean().item(),
            "goal_dist": goal_dist.mean().item(),
            "successes": self.successes.float().mean().item(),
            "consecutive_successes": self.consecutive_successes.item(),
            "act_moving_average": self.act_moving_average,
        }

        # ADR metrics
        if self.adr is not None:
            self.reward_components.update(self.adr.get_metrics())

        self.info_buf["consecutive_successes"] = self.consecutive_successes
        self.info_buf["successes"] = self.successes

    def close(self):
        """Clean up resources."""
        pass

    @property
    def observation_space(self):
        """Gymnasium-compatible observation space."""
        import gymnasium as gym

        return gym.spaces.Box(low=-5.0, high=5.0, shape=(self.num_obs,), dtype=np.float32)

    @property
    def action_space(self):
        """Gymnasium-compatible action space."""
        import gymnasium as gym

        return gym.spaces.Box(low=-1.0, high=1.0, shape=(self.num_actions,), dtype=np.float32)
