"""Kuka iiwa 14 + Allegro Hand Cube Lift Environment.

Replicates the Isaac-Dexsuite-Kuka-Allegro-Lift-v0 task from IsaacLab
using Newton physics engine. The task: reach a cube on a table, grasp it
with the Allegro dexterous hand, and lift it to a target height.

Reference: IsaacLab Newton Physics Integration - Training Environments
"""

from typing import Any

import numpy as np
import torch
import warp as wp

import newton
from newton import JointTargetMode
from newton._src.utils.download_assets import download_git_folder

from .config import EnvConfig

# MuJoCo menagerie URL for Kuka (not in newton-assets yet)
_MENAGERIE_URL = "https://github.com/google-deepmind/mujoco_menagerie.git"


class KukaAllegroLiftEnv:
    """Vectorized environment for Kuka + Allegro cube lift task.

    Observation space (65 dims):
        - Kuka DOF positions (7)
        - Allegro DOF positions (16)
        - Kuka DOF velocities (7)
        - Allegro DOF velocities (16)
        - Cube position (3)
        - Cube orientation (4)
        - Goal position (3)
        - Previous actions (9 = action_dim for direct joint delta)

    Action space (23 dims):
        Direct joint position delta control: 7 Kuka + 16 Allegro.
    """

    def __init__(
        self,
        config: EnvConfig | None = None,
        device: str = "cuda",
        headless: bool = True,
    ):
        self.config = config if config is not None else EnvConfig()
        self.num_envs = self.config.num_envs
        self.device = wp.get_device(device)
        self.torch_device = "cuda" if self.device.is_cuda else "cpu"
        self.headless = headless

        # Timing
        self.fps = self.config.fps
        self.frame_dt = 1.0 / self.fps
        self.sim_substeps = self.config.sim_substeps
        self.sim_dt = self.config.sim_dt_override or (self.frame_dt / self.sim_substeps)

        # Episode
        self.max_episode_length = self.config.episode_length

        # Build simulation
        self._build_simulation()

        # Observation and action spaces
        self.num_actions = self.kuka_dof_count + self.allegro_dof_count  # 23
        self.num_obs = (
            self.kuka_dof_count       # kuka q (7)
            + self.allegro_dof_count  # allegro q (16)
            + self.kuka_dof_count     # kuka qd (7)
            + self.allegro_dof_count  # allegro qd (16)
            + 3                       # cube pos
            + 4                       # cube quat
            + 3                       # goal pos
            + self.num_actions        # prev actions
        )  # = 79

        # RL buffers
        self.obs_buf = torch.zeros(
            self.num_envs, self.num_obs, dtype=torch.float32, device=self.torch_device
        )
        self.reward_buf = torch.zeros(self.num_envs, dtype=torch.float32, device=self.torch_device)
        self.done_buf = torch.zeros(self.num_envs, dtype=torch.bool, device=self.torch_device)
        self.episode_step = torch.zeros(self.num_envs, dtype=torch.int32, device=self.torch_device)

        # Action buffers
        self.actions = torch.zeros(
            self.num_envs, self.num_actions, dtype=torch.float32, device=self.torch_device
        )
        self.prev_actions = torch.zeros_like(self.actions)

        # Joint targets (23D)
        self.current_targets = torch.zeros(
            self.num_envs, self.num_actions, dtype=torch.float32, device=self.torch_device
        )

        # Goal
        self.goal_pos = torch.zeros(self.num_envs, 3, dtype=torch.float32, device=self.torch_device)

        # Success tracking
        self.successes = torch.zeros(self.num_envs, dtype=torch.int32, device=self.torch_device)

        # Reward components for logging
        self.reward_components: dict[str, float] = {}

        # Store initial state
        self._store_initial_state()

    def _build_simulation(self):
        """Build the Newton simulation with Kuka + Allegro, table, and cube."""
        single_env_builder = newton.ModelBuilder()
        newton.solvers.SolverMuJoCo.register_custom_attributes(single_env_builder)

        # Contact parameters
        single_env_builder.default_shape_cfg.ke = 1.0e4
        single_env_builder.default_shape_cfg.kd = 1.0e3
        single_env_builder.default_shape_cfg.mu = 1.2

        # === Load Kuka iiwa 14 from MuJoCo menagerie ===
        kuka_path = download_git_folder(_MENAGERIE_URL, "kuka_iiwa_14")
        robot_pos = wp.vec3(0.0, 0.0, 0.0)
        single_env_builder.add_mjcf(
            str(kuka_path / "iiwa14.xml"),
            xform=wp.transform(robot_pos, wp.quat_identity()),
        )

        self.kuka_body_count = single_env_builder.body_count
        self.kuka_joint_count = single_env_builder.joint_count
        self.kuka_dof_count = single_env_builder.joint_dof_count

        # MJCF doesn't set position control by default - set explicitly
        for i in range(self.kuka_dof_count):
            single_env_builder.joint_target_mode[i] = int(JointTargetMode.POSITION)
            single_env_builder.joint_target_ke[i] = self.config.kuka_stiffness
            single_env_builder.joint_target_kd[i] = self.config.kuka_damping
            single_env_builder.joint_effort_limit[i] = self.config.kuka_effort_limit
            single_env_builder.joint_armature[i] = self.config.kuka_armature

        # Initial Kuka pose: joint0 = -π/2 to face -Y, then elbow bend
        import math
        kuka_init_q = [-math.pi / 2, 0.4, 0.0, -1.4, 0.0, 1.2, 0.0]
        single_env_builder.joint_q[:self.kuka_dof_count] = kuka_init_q
        single_env_builder.joint_target_pos[:self.kuka_dof_count] = kuka_init_q

        # Find end effector body (link7)
        self.ee_body_idx = -1
        for i, key in enumerate(single_env_builder.body_label):
            if "link7" in key:
                self.ee_body_idx = i
                print(f"[INFO] Found EE body: {key} (index {i})")
                break
        if self.ee_body_idx < 0:
            self.ee_body_idx = self.kuka_body_count - 1
            print(f"[INFO] Using fallback EE body index: {self.ee_body_idx}")

        # === Load Allegro hand ===
        allegro_asset = newton.utils.download_asset("wonik_allegro")
        allegro_urdf = allegro_asset / "urdf/allegro_hand_description_left.urdf"

        allegro_body_offset = single_env_builder.body_count
        allegro_joint_offset = single_env_builder.joint_count
        allegro_dof_offset = single_env_builder.joint_dof_count

        # Allegro mounted on Kuka EE
        allegro_xform = wp.transform(
            wp.vec3(0.0, 0.0, 0.1),
            wp.quat_identity(),
        )

        single_env_builder.add_urdf(
            allegro_urdf,
            floating=False,
            xform=allegro_xform,
            enable_self_collisions=False,
        )

        self.allegro_body_offset = allegro_body_offset
        self.allegro_body_count = single_env_builder.body_count - allegro_body_offset
        self.allegro_joint_count = single_env_builder.joint_count - allegro_joint_offset
        self.allegro_dof_count = single_env_builder.joint_dof_count - allegro_dof_offset
        self.allegro_dof_start = allegro_dof_offset

        # Connect Allegro root to Kuka EE
        allegro_root_joint_idx = allegro_joint_offset
        if single_env_builder.joint_parent[allegro_root_joint_idx] == -1:
            single_env_builder.joint_parent[allegro_root_joint_idx] = self.ee_body_idx
            single_env_builder.joint_X_p[allegro_root_joint_idx] = allegro_xform

        # Merge articulations
        for j in range(allegro_joint_offset, single_env_builder.joint_count):
            single_env_builder.joint_articulation[j] = 0
        single_env_builder.articulation_start = [0]
        single_env_builder.articulation_label = ["kuka_allegro"]
        single_env_builder.articulation_world = [0]

        # Set Allegro joint parameters
        for i in range(self.allegro_dof_start, single_env_builder.joint_dof_count):
            single_env_builder.joint_target_ke[i] = self.config.hand_stiffness
            single_env_builder.joint_target_kd[i] = self.config.hand_damping
            single_env_builder.joint_effort_limit[i] = self.config.hand_effort_limit
            single_env_builder.joint_armature[i] = self.config.hand_armature
            single_env_builder.joint_target_mode[i] = int(JointTargetMode.POSITION)

        # Initial Allegro pose (slightly open)
        allegro_init_q = [0.0, 0.3, 0.3, 0.3] * 4
        single_env_builder.joint_q[self.allegro_dof_start:self.allegro_dof_start + self.allegro_dof_count] = allegro_init_q
        single_env_builder.joint_target_pos[self.allegro_dof_start:self.allegro_dof_start + self.allegro_dof_count] = allegro_init_q

        # Store joint limits
        self.kuka_joint_lower = torch.tensor(
            single_env_builder.joint_limit_lower[:self.kuka_dof_count],
            dtype=torch.float32, device=self.torch_device,
        )
        self.kuka_joint_upper = torch.tensor(
            single_env_builder.joint_limit_upper[:self.kuka_dof_count],
            dtype=torch.float32, device=self.torch_device,
        )
        self.allegro_joint_lower = torch.tensor(
            single_env_builder.joint_limit_lower[self.allegro_dof_start:self.allegro_dof_start + self.allegro_dof_count],
            dtype=torch.float32, device=self.torch_device,
        )
        self.allegro_joint_upper = torch.tensor(
            single_env_builder.joint_limit_upper[self.allegro_dof_start:self.allegro_dof_start + self.allegro_dof_count],
            dtype=torch.float32, device=self.torch_device,
        )
        self.allegro_joint_lower = torch.clamp(self.allegro_joint_lower, min=-3.14)
        self.allegro_joint_upper = torch.clamp(self.allegro_joint_upper, max=3.14)

        self.joint_lower = torch.cat([self.kuka_joint_lower, self.allegro_joint_lower])
        self.joint_upper = torch.cat([self.kuka_joint_upper, self.allegro_joint_upper])
        self.joint_range = self.joint_upper - self.joint_lower
        self.joint_mid = (self.joint_upper + self.joint_lower) / 2

        self.total_robot_bodies = single_env_builder.body_count

        # === Add cube as free-floating body ===
        cube_cfg = newton.ModelBuilder.ShapeConfig(
            ke=1.0e4,
            kd=1.0e3,
            mu=1.0,
            density=self.config.cube_mass / (self.config.cube_size ** 3),
        )
        cube_xform = wp.transform(
            wp.vec3(*self.config.cube_spawn_pos),
            wp.quat_identity(),
        )
        cube_body_idx = single_env_builder.add_body(
            xform=cube_xform,
            label="cube",
        )
        single_env_builder.add_shape_box(
            body=cube_body_idx,
            xform=wp.transform_identity(),
            hx=self.config.cube_size / 2,
            hy=self.config.cube_size / 2,
            hz=self.config.cube_size / 2,
            cfg=cube_cfg,
        )
        self.cube_body_idx = cube_body_idx
        self.total_bodies_per_env = single_env_builder.body_count

        # === Replicate for all environments ===
        builder = newton.ModelBuilder()
        newton.solvers.SolverMuJoCo.register_custom_attributes(builder)

        self.env_spacing = (1.5, 1.5, 0.0)
        builder.replicate(single_env_builder, self.num_envs, spacing=self.env_spacing)

        # Ground plane
        builder.default_shape_cfg.ke = 1.0e4
        builder.default_shape_cfg.kd = 1.0e3
        builder.add_ground_plane()

        # === Add table per environment (static shapes not replicated) ===
        table_cfg = newton.ModelBuilder.ShapeConfig(ke=1.0e5, kd=1.0e3, mu=0.8)
        grid_size = int(np.ceil(np.sqrt(self.num_envs)))
        for env_idx in range(self.num_envs):
            grid_x = env_idx // grid_size
            grid_y = env_idx % grid_size
            env_offset_x = grid_x * self.env_spacing[0]
            env_offset_y = grid_y * self.env_spacing[1]

            table_pos = (
                self.config.table_pos[0] + env_offset_x,
                self.config.table_pos[1] + env_offset_y,
                self.config.table_height / 2,
            )
            builder.add_shape_box(
                body=-1,
                xform=wp.transform(wp.vec3(*table_pos), wp.quat_identity()),
                hx=self.config.table_size[0] / 2,
                hy=self.config.table_size[1] / 2,
                hz=self.config.table_size[2] / 2,
                cfg=table_cfg,
            )

        self.model = builder.finalize()

        # Per-env layout
        self.joint_q_per_env = self.model.joint_q.shape[0] // self.num_envs
        self.joint_qd_per_env = self.model.joint_qd.shape[0] // self.num_envs
        self.bodies_per_env = self.model.body_q.shape[0] // self.num_envs

        # Solver
        self.solver = newton.solvers.SolverMuJoCo(
            self.model,
            solver="newton",
            integrator="implicitfast",
            njmax=30 * self.num_envs,
            nconmax=20 * self.num_envs,
            impratio=100.0,
            cone="elliptic",
            iterations=50,
            ls_iterations=100,
            use_mujoco_cpu=False,
        )

        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()
        self.contacts = self.model.collide(self.state_0)

        # Initialize FK
        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.state_0)

        print(f"[INFO] Kuka: {self.kuka_body_count} bodies, {self.kuka_dof_count} DOFs")
        print(f"[INFO] Allegro: {self.allegro_body_count} bodies, {self.allegro_dof_count} DOFs")
        print(f"[INFO] Total per env: {self.total_bodies_per_env} bodies, {self.joint_q_per_env} joint_q")
        print(f"[INFO] Environments: {self.num_envs}")

    def _store_initial_state(self):
        """Store initial state for resetting."""
        initial_q_np = self.state_0.joint_q.numpy().reshape(self.num_envs, self.joint_q_per_env)
        cube_joint_offset = self.joint_q_per_env - 7
        grid_size = int(np.ceil(np.sqrt(self.num_envs)))

        for env_idx in range(self.num_envs):
            grid_x = env_idx // grid_size
            grid_y = env_idx % grid_size
            env_offset_x = grid_x * self.env_spacing[0]
            env_offset_y = grid_y * self.env_spacing[1]

            initial_q_np[env_idx, cube_joint_offset + 0] = self.config.cube_spawn_pos[0] + env_offset_x
            initial_q_np[env_idx, cube_joint_offset + 1] = self.config.cube_spawn_pos[1] + env_offset_y
            initial_q_np[env_idx, cube_joint_offset + 2] = self.config.cube_spawn_pos[2]
            initial_q_np[env_idx, cube_joint_offset + 3:cube_joint_offset + 6] = 0.0
            initial_q_np[env_idx, cube_joint_offset + 6] = 1.0

        self.initial_joint_q = wp.array(initial_q_np.flatten(), dtype=wp.float32, device=self.device)
        self.initial_joint_qd = wp.clone(self.state_0.joint_qd)

        # Recompute FK with corrected cube positions
        self.state_0.joint_q = wp.clone(self.initial_joint_q)
        newton.eval_fk(self.model, self.state_0.joint_q, self.state_0.joint_qd, self.state_0)
        self.initial_body_q = wp.clone(self.state_0.body_q)

        # Initialize targets
        joint_q_np = initial_q_np
        kuka_q = joint_q_np[:, :self.kuka_dof_count]
        allegro_q = joint_q_np[:, self.allegro_dof_start:self.allegro_dof_start + self.allegro_dof_count]
        self.current_targets[:, :self.kuka_dof_count] = torch.from_numpy(kuka_q.copy()).to(self.torch_device)
        self.current_targets[:, self.kuka_dof_count:] = torch.from_numpy(allegro_q.copy()).to(self.torch_device)

    def _simulate_step(self):
        """Run one frame of simulation."""
        self.contacts = self.model.collide(self.state_0)
        for _ in range(self.sim_substeps):
            self.state_0.clear_forces()
            self.solver.step(self.state_0, self.state_1, self.control, self.contacts, self.sim_dt)
            self.state_0, self.state_1 = self.state_1, self.state_0

    def _apply_actions(self, target_pos: torch.Tensor):
        """Apply joint position targets to control buffer."""
        control_np = self.control.joint_target_pos.numpy()
        target_np = target_pos.cpu().numpy()
        control_reshaped = control_np.reshape(self.num_envs, -1)

        control_reshaped[:, :self.kuka_dof_count] = target_np[:, :self.kuka_dof_count]
        control_reshaped[:, self.allegro_dof_start:self.allegro_dof_start + self.allegro_dof_count] = (
            target_np[:, self.kuka_dof_count:]
        )
        self.control.joint_target_pos = wp.array(
            control_reshaped.flatten(), dtype=wp.float32, device=self.device
        )

    def reset(self, env_ids: torch.Tensor | None = None) -> torch.Tensor:
        """Reset specified environments."""
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.torch_device)

        num_reset = len(env_ids)
        if num_reset == 0:
            return self._compute_observations()

        env_ids_np = env_ids.cpu().numpy()

        self.episode_step[env_ids] = 0
        self.successes[env_ids] = 0

        joint_q_np = self.state_0.joint_q.numpy().reshape(self.num_envs, self.joint_q_per_env)
        joint_qd_np = self.state_0.joint_qd.numpy().reshape(self.num_envs, self.joint_qd_per_env)
        initial_q_np = self.initial_joint_q.numpy().reshape(self.num_envs, self.joint_q_per_env)
        initial_qd_np = self.initial_joint_qd.numpy().reshape(self.num_envs, self.joint_qd_per_env)

        noise_q = np.random.uniform(-0.02, 0.02, size=(num_reset, self.joint_q_per_env)).astype(np.float32)
        joint_q_np[env_ids_np] = initial_q_np[env_ids_np] + noise_q
        joint_qd_np[env_ids_np] = initial_qd_np[env_ids_np]

        # Fix cube positions
        cube_joint_offset = self.joint_q_per_env - 7
        cube_qd_offset = self.joint_qd_per_env - 6
        grid_size = int(np.ceil(np.sqrt(self.num_envs)))

        for env_idx in env_ids_np:
            grid_x = env_idx // grid_size
            grid_y = env_idx % grid_size
            env_offset_x = grid_x * self.env_spacing[0]
            env_offset_y = grid_y * self.env_spacing[1]

            cube_x = self.config.cube_spawn_pos[0] + env_offset_x
            cube_y = self.config.cube_spawn_pos[1] + env_offset_y
            cube_z = self.config.cube_spawn_pos[2]

            if self.config.randomize_cube_pos:
                cube_x += np.random.uniform(-self.config.cube_spawn_noise, self.config.cube_spawn_noise)
                cube_y += np.random.uniform(-self.config.cube_spawn_noise, self.config.cube_spawn_noise)

            joint_q_np[env_idx, cube_joint_offset + 0] = cube_x
            joint_q_np[env_idx, cube_joint_offset + 1] = cube_y
            joint_q_np[env_idx, cube_joint_offset + 2] = cube_z
            joint_q_np[env_idx, cube_joint_offset + 3:cube_joint_offset + 6] = 0.0
            joint_q_np[env_idx, cube_joint_offset + 6] = 1.0
            joint_qd_np[env_idx, cube_qd_offset:cube_qd_offset + 6] = 0.0

        self.state_0.joint_q = wp.array(joint_q_np.flatten(), dtype=wp.float32, device=self.device)
        self.state_0.joint_qd = wp.array(joint_qd_np.flatten(), dtype=wp.float32, device=self.device)
        newton.eval_fk(self.model, self.state_0.joint_q, self.state_0.joint_qd, self.state_0)

        # Set goal
        body_q = torch.from_numpy(self.state_0.body_q.numpy()).to(self.torch_device)
        body_q_reshaped = body_q.reshape(self.num_envs, self.bodies_per_env, 7)
        cube_pos = body_q_reshaped[:, self.cube_body_idx, :3]

        self.goal_pos[env_ids, 0] = cube_pos[env_ids, 0]
        self.goal_pos[env_ids, 1] = cube_pos[env_ids, 1]
        self.goal_pos[env_ids, 2] = self.config.table_height + self.config.lift_height

        # Reset action buffers
        self.prev_actions[env_ids] = 0.0
        self.actions[env_ids] = 0.0

        # Reset targets
        joint_q_reshaped = self.state_0.joint_q.numpy().reshape(self.num_envs, self.joint_q_per_env)
        kuka_q = joint_q_reshaped[env_ids_np, :self.kuka_dof_count]
        allegro_q = joint_q_reshaped[env_ids_np, self.allegro_dof_start:self.allegro_dof_start + self.allegro_dof_count]
        self.current_targets[env_ids, :self.kuka_dof_count] = torch.from_numpy(kuka_q.copy()).to(self.torch_device)
        self.current_targets[env_ids, self.kuka_dof_count:] = torch.from_numpy(allegro_q.copy()).to(self.torch_device)

        return self._compute_observations()

    def step(self, actions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, dict]:
        """Take a step in all environments.

        Args:
            actions: Joint position deltas in [-1, 1], shape (num_envs, 23).

        Returns:
            obs, reward, terminated, truncated, info
        """
        self.prev_actions = self.actions.clone()
        self.actions = actions.clone()

        # Relative position control
        action_delta = actions * self.config.action_scale
        self.current_targets = self.current_targets + action_delta
        self.current_targets = torch.clamp(self.current_targets, self.joint_lower, self.joint_upper)

        self._apply_actions(self.current_targets)
        self._simulate_step()

        self.episode_step += 1
        obs = self._compute_observations()
        rewards = self._compute_rewards()
        terminated, truncated = self._compute_dones()

        done_env_ids = torch.where(terminated | truncated)[0]
        if len(done_env_ids) > 0:
            self.reset(done_env_ids)

        return obs, rewards, terminated, truncated, self.reward_components

    def _compute_observations(self) -> torch.Tensor:
        """Compute observations for all environments."""
        joint_q = torch.from_numpy(self.state_0.joint_q.numpy()).to(self.torch_device)
        joint_qd = torch.from_numpy(self.state_0.joint_qd.numpy()).to(self.torch_device)
        body_q = torch.from_numpy(self.state_0.body_q.numpy()).to(self.torch_device)

        joint_q_reshaped = joint_q.reshape(self.num_envs, self.joint_q_per_env)
        joint_qd_reshaped = joint_qd.reshape(self.num_envs, self.joint_qd_per_env)
        body_q_reshaped = body_q.reshape(self.num_envs, self.bodies_per_env, 7)

        # Robot state
        kuka_q = joint_q_reshaped[:, :self.kuka_dof_count]
        kuka_qd = joint_qd_reshaped[:, :self.kuka_dof_count]
        allegro_q = joint_q_reshaped[:, self.allegro_dof_start:self.allegro_dof_start + self.allegro_dof_count]
        allegro_qd = joint_qd_reshaped[:, self.allegro_dof_start:self.allegro_dof_start + self.allegro_dof_count]

        # Cube state
        cube_pos = body_q_reshaped[:, self.cube_body_idx, :3]
        cube_quat = body_q_reshaped[:, self.cube_body_idx, 3:7]

        idx = 0
        self.obs_buf[:, idx:idx + self.kuka_dof_count] = kuka_q
        idx += self.kuka_dof_count
        self.obs_buf[:, idx:idx + self.allegro_dof_count] = allegro_q
        idx += self.allegro_dof_count
        self.obs_buf[:, idx:idx + self.kuka_dof_count] = kuka_qd * 0.1
        idx += self.kuka_dof_count
        self.obs_buf[:, idx:idx + self.allegro_dof_count] = allegro_qd * 0.1
        idx += self.allegro_dof_count
        self.obs_buf[:, idx:idx + 3] = cube_pos
        idx += 3
        self.obs_buf[:, idx:idx + 4] = cube_quat
        idx += 4
        self.obs_buf[:, idx:idx + 3] = self.goal_pos
        idx += 3
        self.obs_buf[:, idx:idx + self.num_actions] = self.prev_actions
        idx += self.num_actions

        self.obs_buf = torch.nan_to_num(self.obs_buf, nan=0.0, posinf=5.0, neginf=-5.0)
        self.obs_buf = torch.clamp(self.obs_buf, -5.0, 5.0)

        return self.obs_buf

    def _compute_rewards(self) -> torch.Tensor:
        """Compute rewards (DEXTRAH-style continuous reward)."""
        body_q = torch.from_numpy(self.state_0.body_q.numpy()).to(self.torch_device)
        body_q_reshaped = body_q.reshape(self.num_envs, self.bodies_per_env, 7)

        ee_pos = body_q_reshaped[:, self.ee_body_idx, :3]
        cube_pos = body_q_reshaped[:, self.cube_body_idx, :3]

        # 1. Hand-to-object distance reward
        hand_to_object_dist = torch.norm(ee_pos - cube_pos, dim=-1)
        hand_to_object_reward = self.config.hand_to_object_weight * torch.exp(
            -self.config.hand_to_object_sharpness * hand_to_object_dist
        )

        # 2. Object-to-goal reward
        object_to_goal_dist = torch.norm(cube_pos - self.goal_pos, dim=-1)
        object_to_goal_reward = self.config.object_to_goal_weight * torch.exp(
            -self.config.object_to_goal_sharpness * object_to_goal_dist
        )

        # 3. Lift reward
        cube_height = cube_pos[:, 2]
        goal_height = self.config.table_height + self.config.lift_height
        vertical_error = torch.abs(cube_height - goal_height)
        lift_reward = self.config.lift_weight * torch.exp(
            -self.config.lift_sharpness * vertical_error
        )

        reward = (hand_to_object_reward + object_to_goal_reward + lift_reward) * self.config.reward_scale
        self.reward_buf = reward

        self.reward_components = {
            "hand_to_object_reward": hand_to_object_reward.mean().item(),
            "object_to_goal_reward": object_to_goal_reward.mean().item(),
            "lift_reward": lift_reward.mean().item(),
            "hand_to_object_dist": hand_to_object_dist.mean().item(),
            "object_to_goal_dist": object_to_goal_dist.mean().item(),
            "cube_height": cube_height.mean().item(),
        }
        return self.reward_buf

    def _compute_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Check termination conditions."""
        body_q = torch.from_numpy(self.state_0.body_q.numpy()).to(self.torch_device)
        body_q_reshaped = body_q.reshape(self.num_envs, self.bodies_per_env, 7)
        cube_pos = body_q_reshaped[:, self.cube_body_idx, :3]

        margin = self.config.workspace_margin
        tx, ty = self.config.table_pos
        tsx, tsy, _ = self.config.table_size
        x_min = tx - tsx / 2 - margin
        x_max = tx + tsx / 2 + margin
        y_min = ty - tsy / 2 - margin
        y_max = ty + tsy / 2 + margin

        out_of_reach = (
            (cube_pos[:, 0] < x_min)
            | (cube_pos[:, 0] > x_max)
            | (cube_pos[:, 1] < y_min)
            | (cube_pos[:, 1] > y_max)
            | (cube_pos[:, 2] < self.config.fall_height)
        )

        timeout = self.episode_step >= self.max_episode_length
        terminated = out_of_reach & ~timeout
        truncated = timeout
        self.done_buf = terminated | truncated

        return terminated, truncated

    def close(self):
        """Clean up resources."""
        pass
