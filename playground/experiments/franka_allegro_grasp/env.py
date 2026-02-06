"""Franka + Allegro Hand Cube Grasping Environment.

This environment combines a Franka arm with an Allegro hand to grasp and lift
a cube from a table. Optionally uses depth sensing for perception.

Reference: DEXTRAH (NVlabs/DEXTRAH) - dextrah_kuka_allegro_env.py
"""

import math
import re
from typing import Any

import numpy as np
import torch
import warp as wp

import newton
from newton import ActuatorMode, Contacts
from newton.sensors import SensorContact, SensorTiledCamera, populate_contacts

from .config import EnvConfig
from .fabric import GraspFabric


class FrankaAllegroGraspEnv:
    """Vectorized environment for Franka + Allegro cube grasping task.

    Task: Reach, grasp, and lift a cube from a table to a target height.

    Observation space:
        - Franka joint positions (7)
        - Franka joint velocities (7)
        - Allegro joint positions (16)
        - Allegro joint velocities (16)
        - End effector position (3)
        - End effector orientation (4)
        - Cube position relative to EE (3)
        - Cube orientation (4)
        - Cube linear velocity (3)
        - Cube angular velocity (3)
        - Goal position (3)
        - Task phase (3) - one-hot [reach, grasp, lift]
        Total: 72 dims (without depth)

        If depth sensor enabled:
        - Depth image (depth_width * depth_height) flattened

    Action space:
        - Franka joint position deltas (7)
        - Allegro joint position deltas (16)
        Total: 23 dims
    """

    def __init__(self, config: EnvConfig, device: str = "cuda", headless: bool = True):
        self.config = config
        self.num_envs = config.num_envs
        self.device = wp.get_device(device)
        self.torch_device = "cuda" if self.device.is_cuda else "cpu"
        self.headless = headless

        # Timing
        self.fps = config.fps
        self.frame_dt = 1.0 / self.fps
        self.sim_substeps = config.sim_substeps
        self.sim_dt = self.frame_dt / self.sim_substeps
        self.control_decimation = config.control_decimation

        # Episode tracking
        self.max_episode_length = config.episode_length
        self.episode_step = torch.zeros(self.num_envs, dtype=torch.int32, device=self.torch_device)

        # Build simulation
        self._build_simulation()

        # Setup depth sensor if enabled
        self.depth_sensor = None
        self.depth_image = None
        if config.use_depth_sensor:
            self._setup_depth_sensor()

        # Observation and action dimensions
        self.num_state_obs = 7 + 7 + 16 + 16 + 3 + 4 + 3 + 4 + 3 + 3 + 3 + 3  # = 72
        self.num_depth_obs = config.depth_width * config.depth_height if config.use_depth_sensor else 0
        self.num_obs = self.num_state_obs + self.num_depth_obs
        self.num_actions = 7 + 16  # Franka (7) + Allegro (16) = 23

        # Buffers
        self.obs_buf = torch.zeros(self.num_envs, self.num_obs, dtype=torch.float32, device=self.torch_device)
        self.reward_buf = torch.zeros(self.num_envs, dtype=torch.float32, device=self.torch_device)
        self.done_buf = torch.zeros(self.num_envs, dtype=torch.bool, device=self.torch_device)
        self.info_buf: dict[str, Any] = {}

        # Action buffers
        self.actions = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float32, device=self.torch_device)
        self.prev_actions = torch.zeros_like(self.actions)

        # Current joint targets
        self.current_targets = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float32, device=self.torch_device)

        # Goal state (lift target position)
        self.goal_pos = torch.zeros(self.num_envs, 3, dtype=torch.float32, device=self.torch_device)

        # Task phase tracking: 0=reach, 1=grasp, 2=lift
        self.task_phase = torch.zeros(self.num_envs, dtype=torch.int32, device=self.torch_device)

        # Success tracking
        self.successes = torch.zeros(self.num_envs, dtype=torch.int32, device=self.torch_device)
        self.consecutive_successes = torch.zeros(self.num_envs, dtype=torch.int32, device=self.torch_device)

        # Reward components for logging
        self.reward_components: dict[str, float] = {}

        # Setup FABRICS module for grasp feature computation
        self.fabric = GraspFabric(
            franka_dof=self.franka_dof_count,
            allegro_dof=self.allegro_dof_count,
            device=self.torch_device,
        )

        # Setup contact sensor for finger-cube contact detection
        self._setup_contact_sensor()

        # Store initial state
        self._store_initial_state()

    def _build_simulation(self):
        """Build the Newton simulation with Franka + Allegro, table, and cube."""
        single_env_builder = newton.ModelBuilder()
        newton.solvers.SolverMuJoCo.register_custom_attributes(single_env_builder)

        # Contact parameters
        single_env_builder.default_shape_cfg.ke = 1.0e4
        single_env_builder.default_shape_cfg.kd = 1.0e3
        single_env_builder.default_shape_cfg.mu = 1.2

        # === Load Franka arm ===
        franka_asset = newton.utils.download_asset("franka_emika_panda")
        # Translate robot so EE is centered on table's Y axis
        # EE naturally points to Y≈-0.75 from base, table center at Y=-0.5
        # Move robot to Y=+0.25 so EE reaches Y≈-0.5
        robot_pos = wp.vec3(0.0, 0.25, 0.0)
        single_env_builder.add_urdf(
            franka_asset / "urdf/fr3.urdf",
            xform=wp.transform(robot_pos, wp.quat_identity()),
            enable_self_collisions=False,
        )

        self.franka_body_count = single_env_builder.body_count
        self.franka_joint_count = single_env_builder.joint_count
        self.franka_dof_count = single_env_builder.joint_dof_count

        # Set Franka joint parameters
        for i in range(self.franka_dof_count):
            single_env_builder.joint_target_ke[i] = self.config.franka_stiffness
            single_env_builder.joint_target_kd[i] = self.config.franka_damping
            single_env_builder.joint_effort_limit[i] = self.config.franka_effort_limit
            single_env_builder.joint_armature[i] = self.config.franka_armature
            single_env_builder.joint_act_mode[i] = int(ActuatorMode.POSITION)

        # Initial Franka pose - arm extended forward
        franka_init_q = [0.0, -0.5, 0.0, -2.0, 0.0, 1.5, 0.785]
        single_env_builder.joint_q[:self.franka_dof_count] = franka_init_q
        single_env_builder.joint_target_pos[:self.franka_dof_count] = franka_init_q

        # Find end effector body
        self.ee_body_idx = -1
        for i, key in enumerate(single_env_builder.body_key):
            if "fr3_link8" in key:
                self.ee_body_idx = i
                break
        if self.ee_body_idx < 0:
            self.ee_body_idx = self.franka_body_count - 1

        # === Load Allegro hand ===
        allegro_asset = newton.utils.download_asset("wonik_allegro")
        allegro_urdf = allegro_asset / "urdf/allegro_hand_description_left.urdf"

        allegro_body_offset = single_env_builder.body_count
        allegro_joint_offset = single_env_builder.joint_count
        allegro_dof_offset = single_env_builder.joint_dof_count

        # Allegro transform relative to Franka EE
        allegro_xform = wp.transform(
            wp.vec3(0.0, 0.0, 0.1),
            wp.quat_from_axis_angle(wp.vec3(0, 0, 1), np.pi),
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

        # Connect Allegro to Franka EE
        allegro_root_joint_idx = allegro_joint_offset
        if single_env_builder.joint_parent[allegro_root_joint_idx] == -1:
            single_env_builder.joint_parent[allegro_root_joint_idx] = self.ee_body_idx
            single_env_builder.joint_X_p[allegro_root_joint_idx] = allegro_xform

        # Merge articulations
        for j in range(allegro_joint_offset, single_env_builder.joint_count):
            single_env_builder.joint_articulation[j] = 0
        single_env_builder.articulation_start = [0]
        single_env_builder.articulation_key = ["franka_allegro"]
        single_env_builder.articulation_world = [0]

        # Set Allegro joint parameters
        for i in range(self.allegro_dof_start, single_env_builder.joint_dof_count):
            single_env_builder.joint_target_ke[i] = self.config.hand_stiffness
            single_env_builder.joint_target_kd[i] = self.config.hand_damping
            single_env_builder.joint_effort_limit[i] = self.config.hand_effort_limit
            single_env_builder.joint_armature[i] = self.config.hand_armature
            single_env_builder.joint_act_mode[i] = int(ActuatorMode.POSITION)

        # Initial Allegro pose (slightly open)
        allegro_init_q = [0.0, 0.3, 0.3, 0.3] * 4  # 4 fingers
        single_env_builder.joint_q[self.allegro_dof_start:self.allegro_dof_start + self.allegro_dof_count] = allegro_init_q
        single_env_builder.joint_target_pos[self.allegro_dof_start:self.allegro_dof_start + self.allegro_dof_count] = allegro_init_q

        # Store joint limits
        self.franka_joint_lower = torch.tensor(
            single_env_builder.joint_limit_lower[:self.franka_dof_count],
            dtype=torch.float32, device=self.torch_device
        )
        self.franka_joint_upper = torch.tensor(
            single_env_builder.joint_limit_upper[:self.franka_dof_count],
            dtype=torch.float32, device=self.torch_device
        )
        self.allegro_joint_lower = torch.tensor(
            single_env_builder.joint_limit_lower[self.allegro_dof_start:self.allegro_dof_start + self.allegro_dof_count],
            dtype=torch.float32, device=self.torch_device
        )
        self.allegro_joint_upper = torch.tensor(
            single_env_builder.joint_limit_upper[self.allegro_dof_start:self.allegro_dof_start + self.allegro_dof_count],
            dtype=torch.float32, device=self.torch_device
        )

        # Clamp extreme values
        self.allegro_joint_lower = torch.clamp(self.allegro_joint_lower, min=-3.14)
        self.allegro_joint_upper = torch.clamp(self.allegro_joint_upper, max=3.14)

        # Joint range for normalization
        self.joint_lower = torch.cat([self.franka_joint_lower, self.allegro_joint_lower])
        self.joint_upper = torch.cat([self.franka_joint_upper, self.allegro_joint_upper])
        self.joint_range = self.joint_upper - self.joint_lower
        self.joint_mid = (self.joint_upper + self.joint_lower) / 2

        self.total_robot_bodies = single_env_builder.body_count

        # NOTE: Table will be added after replication (static shapes on body=-1 are not replicated)
        # Cube is added here but positions will be corrected in reset()

        # === Add cube as free-floating body ===
        cube_cfg = newton.ModelBuilder.ShapeConfig(
            ke=1.0e4,
            kd=1.0e3,
            mu=1.0,
            density=self.config.cube_mass / (self.config.cube_size ** 3),
        )
        # Initial position (will be corrected per-environment in reset)
        cube_xform = wp.transform(
            wp.vec3(*self.config.cube_spawn_pos),
            wp.quat_identity(),
        )
        cube_body_idx = single_env_builder.add_body(
            xform=cube_xform,
            key="cube",
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

        # Spacing for environment grid layout
        self.env_spacing = (1.5, 1.5, 0.0)
        builder.replicate(single_env_builder, self.num_envs, spacing=self.env_spacing)

        # Add ground plane
        builder.default_shape_cfg.ke = 1.0e4
        builder.default_shape_cfg.kd = 1.0e3
        builder.add_ground_plane()

        # === Add table for each environment (static shapes must be added after replication) ===
        table_cfg = newton.ModelBuilder.ShapeConfig(
            ke=1.0e5,
            kd=1.0e3,
            mu=0.8,
        )
        # Calculate grid layout (same as replicate uses)
        grid_size = int(np.ceil(np.sqrt(self.num_envs)))
        for env_idx in range(self.num_envs):
            # Grid position
            grid_x = env_idx // grid_size
            grid_y = env_idx % grid_size
            env_offset_x = grid_x * self.env_spacing[0]
            env_offset_y = grid_y * self.env_spacing[1]

            # Table position for this environment (using config values)
            table_pos = (
                self.config.table_pos[0] + env_offset_x,
                self.config.table_pos[1] + env_offset_y,
                self.config.table_height / 2,
            )
            table_xform = wp.transform(wp.vec3(*table_pos), wp.quat_identity())

            builder.add_shape_box(
                body=-1,  # Static (world body)
                xform=table_xform,
                hx=self.config.table_size[0] / 2,
                hy=self.config.table_size[1] / 2,
                hz=self.config.table_size[2] / 2,
                cfg=table_cfg,
            )

        self.model = builder.finalize()

        # Calculate actual sizes per environment
        self.joint_q_per_env = self.model.joint_q.shape[0] // self.num_envs
        self.joint_qd_per_env = self.model.joint_qd.shape[0] // self.num_envs
        self.bodies_per_env = self.model.body_q.shape[0] // self.num_envs

        # Solver
        self.solver = newton.solvers.SolverMuJoCo(
            self.model,
            solver="newton",
            integrator="implicitfast",
            njmax=500 * self.num_envs,
            nconmax=300 * self.num_envs,
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

    def _setup_depth_sensor(self):
        """Setup depth camera sensor."""
        # New API: width/height/num_cameras moved from constructor to create_*_output methods
        self.depth_sensor = SensorTiledCamera(
            model=self.model,
            options=SensorTiledCamera.Options(
                default_light=True,
                colors_per_shape=True,
            ),
        )

        # Compute camera rays (new API: width, height as arguments)
        fov_rad = math.radians(self.config.depth_fov)
        self.camera_rays = self.depth_sensor.compute_pinhole_camera_rays(
            self.config.depth_width, self.config.depth_height, fov_rad
        )

        # Create depth output buffer (new API: width, height, num_cameras as arguments)
        # Output shape is now (num_worlds, num_cameras, height, width)
        self.depth_image = self.depth_sensor.create_depth_image_output(
            self.config.depth_width, self.config.depth_height, num_cameras=1
        )

        # Camera transforms will be updated based on EE pose
        self.camera_transforms = wp.zeros((self.num_envs, 1), dtype=wp.transformf, device=self.device)

    def _setup_contact_sensor(self):
        """Setup contact sensor for finger-cube contact detection."""
        # Create a Contacts object for storing contact info
        self.contact_data = Contacts(0, 0)

        # Find fingertip shapes (link3 are the distal links in Allegro)
        # Allegro finger structure: link0 (base) -> link1 -> link2 -> link3 (tip)
        # We want to detect contacts between fingertip shapes and the cube
        try:
            self.finger_contact_sensor = SensorContact(
                self.model,
                sensing_obj_shapes=".*link3.*",  # Fingertip links
                counterpart_shapes=".*cube.*",   # Cube
                match_fn=lambda string, pat: re.match(pat, string, re.IGNORECASE),
                include_total=True,
                verbose=False,
            )
            self.use_contact_sensor = True
        except Exception as e:
            print(f"[WARNING] Failed to setup contact sensor: {e}")
            self.finger_contact_sensor = None
            self.use_contact_sensor = False

    def _store_initial_state(self):
        """Store initial state for resetting environments."""
        self.initial_joint_q = wp.clone(self.state_0.joint_q)
        self.initial_joint_qd = wp.clone(self.state_0.joint_qd)
        self.initial_body_q = wp.clone(self.state_0.body_q)
        self.initial_body_qd = wp.clone(self.state_0.body_qd)

        # Correct initial cube positions in initial_joint_q
        # The replicated cube positions may not be correct, so we fix them here
        initial_q_np = self.initial_joint_q.numpy().reshape(self.num_envs, self.joint_q_per_env)
        cube_joint_offset = self.joint_q_per_env - 7
        grid_size = int(np.ceil(np.sqrt(self.num_envs)))

        for env_idx in range(self.num_envs):
            # Calculate environment offset in grid layout
            grid_x = env_idx // grid_size
            grid_y = env_idx % grid_size
            env_offset_x = grid_x * self.env_spacing[0]
            env_offset_y = grid_y * self.env_spacing[1]

            # Set correct cube world position (above table)
            initial_q_np[env_idx, cube_joint_offset + 0] = self.config.cube_spawn_pos[0] + env_offset_x
            initial_q_np[env_idx, cube_joint_offset + 1] = self.config.cube_spawn_pos[1] + env_offset_y
            initial_q_np[env_idx, cube_joint_offset + 2] = self.config.cube_spawn_pos[2]
            # Identity quaternion
            initial_q_np[env_idx, cube_joint_offset + 3] = 0.0
            initial_q_np[env_idx, cube_joint_offset + 4] = 0.0
            initial_q_np[env_idx, cube_joint_offset + 5] = 0.0
            initial_q_np[env_idx, cube_joint_offset + 6] = 1.0

        self.initial_joint_q = wp.array(initial_q_np.flatten(), dtype=wp.float32, device=self.device)

        # Update state_0 with corrected cube positions and run FK
        self.state_0.joint_q = wp.clone(self.initial_joint_q)
        newton.eval_fk(self.model, self.state_0.joint_q, self.state_0.joint_qd, self.state_0)

        # Update initial_body_q with corrected positions
        self.initial_body_q = wp.clone(self.state_0.body_q)

        # Initialize current targets
        joint_q_np = initial_q_np

        # Extract robot joint positions (Franka + Allegro)
        franka_q = joint_q_np[:, :self.franka_dof_count]
        allegro_q = joint_q_np[:, self.allegro_dof_start:self.allegro_dof_start + self.allegro_dof_count]

        self.current_targets[:, :self.franka_dof_count] = torch.from_numpy(franka_q.copy()).to(self.torch_device)
        self.current_targets[:, self.franka_dof_count:] = torch.from_numpy(allegro_q.copy()).to(self.torch_device)

    def _simulate_step(self):
        """Run one frame of simulation."""
        self.contacts = self.model.collide(self.state_0)

        for _ in range(self.sim_substeps):
            self.state_0.clear_forces()
            self.solver.step(self.state_0, self.state_1, self.control, self.contacts, self.sim_dt)
            self.state_0, self.state_1 = self.state_1, self.state_0

    def reset(self, env_ids: torch.Tensor | None = None) -> torch.Tensor:
        """Reset specified environments."""
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.torch_device)

        num_reset = len(env_ids)
        if num_reset == 0:
            return self._compute_observations()

        env_ids_np = env_ids.cpu().numpy()

        # Reset episode counters
        self.episode_step[env_ids] = 0
        self.successes[env_ids] = 0
        self.consecutive_successes[env_ids] = 0
        self.task_phase[env_ids] = 0  # Start in reach phase

        # Reset joint states
        joint_q_np = self.state_0.joint_q.numpy().reshape(self.num_envs, self.joint_q_per_env)
        joint_qd_np = self.state_0.joint_qd.numpy().reshape(self.num_envs, self.joint_qd_per_env)
        initial_q_np = self.initial_joint_q.numpy().reshape(self.num_envs, self.joint_q_per_env)
        initial_qd_np = self.initial_joint_qd.numpy().reshape(self.num_envs, self.joint_qd_per_env)

        # Reset robot joints with small noise
        noise_q = np.random.uniform(-0.02, 0.02, size=(num_reset, self.joint_q_per_env)).astype(np.float32)
        joint_q_np[env_ids_np] = initial_q_np[env_ids_np] + noise_q
        joint_qd_np[env_ids_np] = initial_qd_np[env_ids_np]

        # === Set correct cube positions for each environment ===
        # The cube has a free joint with 7 DOFs: [px, py, pz, qx, qy, qz, qw]
        cube_joint_offset = self.joint_q_per_env - 7
        grid_size = int(np.ceil(np.sqrt(self.num_envs)))

        for env_idx in env_ids_np:
            # Calculate environment offset in grid layout
            grid_x = env_idx // grid_size
            grid_y = env_idx % grid_size
            env_offset_x = grid_x * self.env_spacing[0]
            env_offset_y = grid_y * self.env_spacing[1]

            # Calculate correct cube world position (above table)
            cube_x = self.config.cube_spawn_pos[0] + env_offset_x
            cube_y = self.config.cube_spawn_pos[1] + env_offset_y
            cube_z = self.config.cube_spawn_pos[2]

            # Add optional noise
            if self.config.randomize_cube_pos:
                cube_x += np.random.uniform(-self.config.cube_spawn_noise, self.config.cube_spawn_noise)
                cube_y += np.random.uniform(-self.config.cube_spawn_noise, self.config.cube_spawn_noise)

            # Set cube position in joint_q
            joint_q_np[env_idx, cube_joint_offset + 0] = cube_x
            joint_q_np[env_idx, cube_joint_offset + 1] = cube_y
            joint_q_np[env_idx, cube_joint_offset + 2] = cube_z
            # Set identity quaternion
            joint_q_np[env_idx, cube_joint_offset + 3] = 0.0  # qx
            joint_q_np[env_idx, cube_joint_offset + 4] = 0.0  # qy
            joint_q_np[env_idx, cube_joint_offset + 5] = 0.0  # qz
            joint_q_np[env_idx, cube_joint_offset + 6] = 1.0  # qw

        # Reset cube velocities to zero
        cube_qd_offset = self.joint_qd_per_env - 6  # Free joint has 6 velocity DOFs
        for env_idx in env_ids_np:
            joint_qd_np[env_idx, cube_qd_offset:cube_qd_offset + 6] = 0.0

        self.state_0.joint_q = wp.array(joint_q_np.flatten(), dtype=wp.float32, device=self.device)
        self.state_0.joint_qd = wp.array(joint_qd_np.flatten(), dtype=wp.float32, device=self.device)

        # Recompute FK
        newton.eval_fk(self.model, self.state_0.joint_q, self.state_0.joint_qd, self.state_0)

        # Set goal position (lift target above cube spawn position)
        body_q = torch.from_numpy(self.state_0.body_q.numpy()).to(self.torch_device)
        body_q_reshaped = body_q.reshape(self.num_envs, self.bodies_per_env, 7)
        cube_pos = body_q_reshaped[:, self.cube_body_idx, :3]

        self.goal_pos[env_ids, 0] = cube_pos[env_ids, 0]
        self.goal_pos[env_ids, 1] = cube_pos[env_ids, 1]
        self.goal_pos[env_ids, 2] = self.config.table_height + self.config.lift_height

        # Reset action buffers
        self.prev_actions[env_ids] = 0.0
        self.actions[env_ids] = 0.0

        # Reset current targets
        joint_q_reshaped = self.state_0.joint_q.numpy().reshape(self.num_envs, self.joint_q_per_env)
        franka_q = joint_q_reshaped[env_ids_np, :self.franka_dof_count]
        allegro_q = joint_q_reshaped[env_ids_np, self.allegro_dof_start:self.allegro_dof_start + self.allegro_dof_count]

        self.current_targets[env_ids, :self.franka_dof_count] = torch.from_numpy(franka_q.copy()).to(self.torch_device)
        self.current_targets[env_ids, self.franka_dof_count:] = torch.from_numpy(allegro_q.copy()).to(self.torch_device)

        return self._compute_observations()

    def step(self, actions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict]:
        """Take a step in all environments."""
        self.prev_actions = self.actions.clone()
        self.actions = actions.clone()

        # Apply delta actions
        if self.config.use_relative_control:
            action_delta = actions * self.config.action_scale
            self.current_targets = self.current_targets + action_delta
        else:
            self.current_targets = self.joint_mid + actions * (self.joint_range / 2) * self.config.action_scale

        # Clamp to joint limits
        self.current_targets = torch.clamp(self.current_targets, self.joint_lower, self.joint_upper)

        # Apply to control
        self._apply_actions(self.current_targets)

        # Step simulation
        for _ in range(self.control_decimation):
            self._simulate_step()

        # Update episode step
        self.episode_step += 1

        # Compute observations
        obs = self._compute_observations()

        # Compute rewards
        rewards = self._compute_rewards()

        # Check termination
        dones = self._compute_dones()

        # Auto-reset done environments
        done_env_ids = torch.where(dones)[0]
        if len(done_env_ids) > 0:
            self.reset(done_env_ids)

        return obs, rewards, dones, self.info_buf

    def _apply_actions(self, target_pos: torch.Tensor):
        """Apply joint position targets to control buffer."""
        control_np = self.control.joint_target_pos.numpy()
        target_np = target_pos.cpu().numpy()

        # Reshape control for vectorized assignment
        control_reshaped = control_np.reshape(self.num_envs, -1)

        # Apply Franka targets
        control_reshaped[:, :self.franka_dof_count] = target_np[:, :self.franka_dof_count]

        # Apply Allegro targets
        control_reshaped[:, self.allegro_dof_start:self.allegro_dof_start + self.allegro_dof_count] = target_np[:, self.franka_dof_count:]

        self.control.joint_target_pos = wp.array(control_reshaped.flatten(), dtype=wp.float32, device=self.device)

    def _compute_observations(self) -> torch.Tensor:
        """Compute observations for all environments."""
        joint_q = torch.from_numpy(self.state_0.joint_q.numpy()).to(self.torch_device)
        joint_qd = torch.from_numpy(self.state_0.joint_qd.numpy()).to(self.torch_device)
        body_q = torch.from_numpy(self.state_0.body_q.numpy()).to(self.torch_device)
        body_qd = torch.from_numpy(self.state_0.body_qd.numpy()).to(self.torch_device)

        # Reshape
        joint_q_reshaped = joint_q.reshape(self.num_envs, self.joint_q_per_env)
        joint_qd_reshaped = joint_qd.reshape(self.num_envs, self.joint_qd_per_env)
        body_q_reshaped = body_q.reshape(self.num_envs, self.bodies_per_env, 7)
        body_qd_reshaped = body_qd.reshape(self.num_envs, self.bodies_per_env, 6)

        idx = 0

        # Franka joint positions (normalized)
        franka_q = joint_q_reshaped[:, :self.franka_dof_count]
        franka_q_norm = (franka_q - self.franka_joint_lower) / (self.franka_joint_upper - self.franka_joint_lower + 1e-6) * 2 - 1
        self.obs_buf[:, idx:idx + 7] = franka_q_norm
        idx += 7

        # Franka joint velocities (scaled)
        franka_qd = joint_qd_reshaped[:, :self.franka_dof_count]
        self.obs_buf[:, idx:idx + 7] = franka_qd * 0.1
        idx += 7

        # Allegro joint positions (normalized)
        allegro_q = joint_q_reshaped[:, self.allegro_dof_start:self.allegro_dof_start + self.allegro_dof_count]
        allegro_q_norm = (allegro_q - self.allegro_joint_lower) / (self.allegro_joint_upper - self.allegro_joint_lower + 1e-6) * 2 - 1
        self.obs_buf[:, idx:idx + 16] = allegro_q_norm
        idx += 16

        # Allegro joint velocities (scaled)
        allegro_qd = joint_qd_reshaped[:, self.allegro_dof_start:self.allegro_dof_start + self.allegro_dof_count]
        self.obs_buf[:, idx:idx + 16] = allegro_qd * 0.1
        idx += 16

        # End effector position
        ee_pos = body_q_reshaped[:, self.ee_body_idx, :3]
        self.obs_buf[:, idx:idx + 3] = ee_pos
        idx += 3

        # End effector orientation
        ee_quat = body_q_reshaped[:, self.ee_body_idx, 3:7]
        self.obs_buf[:, idx:idx + 4] = ee_quat
        idx += 4

        # Cube position relative to EE
        cube_pos = body_q_reshaped[:, self.cube_body_idx, :3]
        self.obs_buf[:, idx:idx + 3] = cube_pos - ee_pos
        idx += 3

        # Cube orientation
        cube_quat = body_q_reshaped[:, self.cube_body_idx, 3:7]
        self.obs_buf[:, idx:idx + 4] = cube_quat
        idx += 4

        # Cube velocities
        cube_vel = body_qd_reshaped[:, self.cube_body_idx, :3]
        cube_ang_vel = body_qd_reshaped[:, self.cube_body_idx, 3:6]
        self.obs_buf[:, idx:idx + 3] = cube_vel * 0.1
        idx += 3
        self.obs_buf[:, idx:idx + 3] = cube_ang_vel * 0.1
        idx += 3

        # Goal position
        self.obs_buf[:, idx:idx + 3] = self.goal_pos
        idx += 3

        # Task phase (one-hot)
        phase_onehot = torch.zeros(self.num_envs, 3, device=self.torch_device)
        phase_onehot.scatter_(1, self.task_phase.unsqueeze(1), 1.0)
        self.obs_buf[:, idx:idx + 3] = phase_onehot
        idx += 3

        # Depth image if enabled
        if self.config.use_depth_sensor and self.depth_sensor is not None:
            self._update_depth_sensor()
            # New output shape: (num_worlds, num_cameras, height, width)
            depth_np = self.depth_image.numpy()
            # Flatten to (num_envs, height * width) for observation
            depth_flat = depth_np.reshape(self.num_envs, -1)

            # Normalize depth
            depth_normalized = (depth_flat - self.config.depth_min) / (self.config.depth_max - self.config.depth_min)
            depth_normalized = np.clip(depth_normalized, 0.0, 1.0)

            self.obs_buf[:, idx:idx + self.num_depth_obs] = torch.from_numpy(depth_normalized).to(self.torch_device)

        # Handle NaN
        self.obs_buf = torch.nan_to_num(self.obs_buf, nan=0.0, posinf=5.0, neginf=-5.0)
        self.obs_buf = torch.clamp(self.obs_buf, -5.0, 5.0)

        return self.obs_buf

    def _update_depth_sensor(self):
        """Update depth sensor based on current EE pose."""
        if self.depth_sensor is None:
            return

        # Fixed camera position: at the short side of table, opposite from before (Y = +0.1)
        # Table center at (-0.3, -0.5), table extends Y: [-0.9, -0.1]
        # Camera placed at Y = 0.1 (beyond the near Y edge), looking toward table
        grid_size = int(np.ceil(np.sqrt(self.num_envs)))

        transforms_list = []
        for i in range(self.num_envs):
            # Calculate environment offset
            grid_x = i // grid_size
            grid_y = i % grid_size
            env_offset_x = grid_x * self.env_spacing[0]
            env_offset_y = grid_y * self.env_spacing[1]

            # Camera position: opposite short side of table (near robot side), lower height
            cam_x = -0.3 + env_offset_x   # Centered on table X
            cam_y = 0.1 + env_offset_y    # Beyond the near Y edge of table (opposite from -1.1)
            cam_z = 0.43                   # Lower height (closer to table level)

            # Look toward the table/cube area
            target_x = -0.3 + env_offset_x
            target_y = -0.5 + env_offset_y
            target_z = 0.45  # Cube height

            # Compute look-at quaternion
            # Camera forward is -Z in camera space, we want it to point at target
            cam_pos = np.array([cam_x, cam_y, cam_z])
            target_pos = np.array([target_x, target_y, target_z])
            forward = target_pos - cam_pos
            forward = forward / np.linalg.norm(forward)

            # Compute rotation from -Z to forward direction
            # Using axis-angle: axis = cross(-Z, forward), angle = acos(dot(-Z, forward))
            neg_z = np.array([0.0, 0.0, -1.0], dtype=np.float32)
            axis = np.cross(neg_z, forward).astype(np.float32)
            axis_norm = np.linalg.norm(axis)
            if axis_norm > 1e-6:
                axis = axis / axis_norm
                angle = float(np.arccos(np.clip(np.dot(neg_z, forward), -1.0, 1.0)))
                quat = wp.quat_from_axis_angle(wp.vec3(axis[0], axis[1], axis[2]), angle)
            else:
                quat = wp.quat_identity()

            transforms_list.append([wp.transform(
                wp.vec3(cam_x, cam_y, cam_z),
                quat
            )])

        # Create 2D warp array (num_envs, 1)
        self.camera_transforms = wp.array(transforms_list, dtype=wp.transformf, device=self.device)

        # Render depth
        self.depth_sensor.render(
            state=self.state_0,
            camera_transforms=self.camera_transforms,
            camera_rays=self.camera_rays,
            depth_image=self.depth_image,
        )

    def _compute_rewards(self) -> torch.Tensor:
        """Compute rewards using DEXTRAH continuous reward structure.

        DEXTRAH reward = hand_to_object + object_to_goal + finger_curl_reg + lift
        All components are computed continuously and summed (not phase-based).
        """
        body_q = torch.from_numpy(self.state_0.body_q.numpy()).to(self.torch_device)
        joint_q = torch.from_numpy(self.state_0.joint_q.numpy()).to(self.torch_device)
        joint_qd = torch.from_numpy(self.state_0.joint_qd.numpy()).to(self.torch_device)

        body_q_reshaped = body_q.reshape(self.num_envs, self.bodies_per_env, 7)
        joint_q_reshaped = joint_q.reshape(self.num_envs, self.joint_q_per_env)
        joint_qd_reshaped = joint_qd.reshape(self.num_envs, self.joint_qd_per_env)

        # Get positions
        ee_pos = body_q_reshaped[:, self.ee_body_idx, :3]
        ee_quat = body_q_reshaped[:, self.ee_body_idx, 3:7]
        cube_pos = body_q_reshaped[:, self.cube_body_idx, :3]

        # Get Allegro joint positions
        allegro_q = joint_q_reshaped[
            :, self.allegro_dof_start:self.allegro_dof_start + self.allegro_dof_count
        ]

        # === DEXTRAH Reward Components (continuous, all summed) ===

        # 1. Hand-to-Object Reward: weight * exp(-sharpness * dist)
        # Encourages fingertip and palm to approach object
        hand_to_object_dist = torch.norm(ee_pos - cube_pos, dim=-1)
        hand_to_object_reward = self.config.hand_to_object_weight * torch.exp(
            -self.config.hand_to_object_sharpness * hand_to_object_dist
        )

        # 2. Object-to-Goal Reward: weight * exp(-sharpness * dist)
        # Incentivizes moving object toward goal position
        object_to_goal_dist = torch.norm(cube_pos - self.goal_pos, dim=-1)
        object_to_goal_reward = self.config.object_to_goal_weight * torch.exp(
            -self.config.object_to_goal_sharpness * object_to_goal_dist
        )

        # 3. Finger Curl Regularization: weight * ||q - curled_q||²
        # Prevents excessive finger curling during approach (negative weight = penalty)
        # Curled configuration: nominally closed fingers
        curled_q = torch.tensor(
            [0.0, 0.8, 0.8, 0.8] * 4,  # 4 fingers, each with [abduction, proximal, middle, distal]
            dtype=torch.float32, device=self.torch_device
        ).unsqueeze(0).expand(self.num_envs, -1)
        finger_curl_diff = allegro_q - curled_q
        finger_curl_reg = self.config.finger_curl_reg_weight * torch.sum(finger_curl_diff ** 2, dim=-1)

        # 4. Lift Reward: weight * exp(-sharpness * vertical_error)
        # Rewards lifting object off table toward goal height
        cube_height = cube_pos[:, 2]
        goal_height = self.config.table_height + self.config.lift_height
        vertical_error = torch.abs(cube_height - goal_height)
        lift_reward = self.config.lift_weight * torch.exp(
            -self.config.lift_sharpness * vertical_error
        )

        # 5. In-Success-Region Bonus
        # Bonus when object is at goal position
        in_success_region = object_to_goal_dist < self.config.object_goal_tol
        success_bonus = torch.where(
            in_success_region,
            torch.full_like(hand_to_object_reward, self.config.in_success_region_weight),
            torch.zeros_like(hand_to_object_reward)
        )

        # === Compute FABRICS grasp features (for logging and optional use) ===
        grasp_features = self.fabric.compute_grasp_features(
            ee_pos, ee_quat, allegro_q, cube_pos, self.config.cube_size
        )
        fabric_rewards = self.fabric.compute_grasp_reward(
            grasp_features, self.config.cube_size
        )

        # === DEXTRAH Total Reward (continuous sum) ===
        reward = (
            hand_to_object_reward +
            object_to_goal_reward +
            finger_curl_reg +  # This is negative (penalty)
            lift_reward +
            success_bonus
        )

        # === Additional Penalties ===

        # Action penalty
        action_penalty = self.config.action_penalty_scale * torch.sum(self.actions ** 2, dim=-1)

        # Action delta penalty
        action_delta = self.actions - self.prev_actions
        action_delta_penalty = self.config.action_delta_penalty_scale * torch.sum(action_delta ** 2, dim=-1)

        # Velocity penalty
        vel_penalty = self.config.velocity_penalty_scale * torch.sum(joint_qd_reshaped ** 2, dim=-1)

        # Drop penalty
        dropped = cube_height < self.config.fall_height
        drop_penalty = torch.where(dropped, torch.full_like(reward, self.config.drop_penalty), torch.zeros_like(reward))

        # === Update task phase (for logging/visualization) ===
        reached = hand_to_object_dist < self.config.hand_to_object_dist_threshold
        self.task_phase = torch.where(
            (self.task_phase == 0) & reached,
            torch.ones_like(self.task_phase),
            self.task_phase
        )
        grasped = reached & (cube_height > self.config.table_height + 0.05)
        self.task_phase = torch.where(
            (self.task_phase == 1) & grasped,
            torch.full_like(self.task_phase, 2),
            self.task_phase
        )

        # Success tracking
        is_success = in_success_region & (cube_height > self.config.table_height + self.config.object_height_thresh)
        self.consecutive_successes = torch.where(is_success, self.consecutive_successes + 1, torch.zeros_like(self.consecutive_successes))
        self.successes = torch.where(
            self.consecutive_successes >= self.config.consecutive_successes,
            self.successes + 1,
            self.successes
        )

        # Total reward
        self.reward_buf = reward - action_penalty - action_delta_penalty - vel_penalty + drop_penalty

        # Store components for logging
        phase_0_mask = self.task_phase == 0
        phase_1_mask = self.task_phase == 1
        phase_2_mask = self.task_phase == 2

        self.reward_components = {
            # DEXTRAH reward components
            "hand_to_object_reward": hand_to_object_reward.mean().item(),
            "object_to_goal_reward": object_to_goal_reward.mean().item(),
            "finger_curl_reg": finger_curl_reg.mean().item(),
            "lift_reward": lift_reward.mean().item(),
            "success_bonus": success_bonus.mean().item(),
            # Penalties
            "action_penalty": -action_penalty.mean().item(),
            "drop_penalty": drop_penalty.mean().item(),
            # Distances
            "hand_to_object_dist": hand_to_object_dist.mean().item(),
            "object_to_goal_dist": object_to_goal_dist.mean().item(),
            "cube_height": cube_height.mean().item(),
            "vertical_error": vertical_error.mean().item(),
            # Phase tracking (for visualization)
            "phase_reach": phase_0_mask.float().mean().item(),
            "phase_grasp": phase_1_mask.float().mean().item(),
            "phase_lift": phase_2_mask.float().mean().item(),
            # FABRICS metrics (for comparison)
            "fabric_contact": fabric_rewards["contact_reward"].mean().item(),
            "fabric_closure": fabric_rewards["closure_reward"].mean().item(),
            "fabric_approach": fabric_rewards["approach_reward"].mean().item(),
            "grasp_closure_dist": grasp_features["grasp_closure"].mean().item(),
        }

        return self.reward_buf

    def _compute_dones(self) -> torch.Tensor:
        """Check termination conditions."""
        body_q = torch.from_numpy(self.state_0.body_q.numpy()).to(self.torch_device)
        body_q_reshaped = body_q.reshape(self.num_envs, self.bodies_per_env, 7)

        cube_pos = body_q_reshaped[:, self.cube_body_idx, :3]

        # Termination conditions
        timeout = self.episode_step >= self.max_episode_length
        cube_fell = cube_pos[:, 2] < self.config.fall_height - 0.1

        self.done_buf = timeout | cube_fell

        return self.done_buf

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
