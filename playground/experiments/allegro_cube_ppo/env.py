"""Allegro Hand Cube Rotation Environment for RL training with Newton.

Reference: IsaacGymEnvs DextrEme (allegro_hand_dextreme.py)
"""

from typing import Any

import numpy as np
import torch
import warp as wp

import newton
from newton import ActuatorMode

from .config import EnvConfig


class AllegroHandCubeEnv:
    """Vectorized environment for Allegro Hand cube rotation task (DextrEme style).

    Observation space (49 dims):
        - Hand joint positions normalized (16)
        - Hand joint velocities scaled (16)
        - Cube position relative to palm (3)
        - Cube orientation quaternion (4)
        - Cube linear velocity (3)
        - Cube angular velocity (3)
        - Goal orientation quaternion (4)

    Action space (16 dims):
        - Joint position deltas for 16 actuated DOFs
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

        # Observation and action spaces
        self.num_obs = 16 + 16 + 3 + 4 + 3 + 3 + 4  # = 49
        self.num_actions = 16

        # Buffers
        self.obs_buf = torch.zeros(self.num_envs, self.num_obs, dtype=torch.float32, device=self.torch_device)
        self.reward_buf = torch.zeros(self.num_envs, dtype=torch.float32, device=self.torch_device)
        self.done_buf = torch.zeros(self.num_envs, dtype=torch.bool, device=self.torch_device)
        self.info_buf: dict[str, Any] = {}

        # Action buffers
        self.actions = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float32, device=self.torch_device)
        self.prev_actions = torch.zeros_like(self.actions)

        # Current joint targets (for delta actions)
        self.current_targets = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float32, device=self.torch_device)

        # Goal state (quaternion: x, y, z, w)
        self.goal_quat = torch.zeros(self.num_envs, 4, dtype=torch.float32, device=self.torch_device)
        self.goal_quat[:, 3] = 1.0  # identity

        # Success tracking (DextrEme: consecutive successes)
        self.successes = torch.zeros(self.num_envs, dtype=torch.int32, device=self.torch_device)
        self.consecutive_successes = torch.zeros(self.num_envs, dtype=torch.int32, device=self.torch_device)

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
            # Note: Don't ignore DexCube/visuals so cube is visible
            ignore_paths=[".*Dummy", ".*CollisionPlane", ".*goal"],
        )

        self.num_hand_dofs = hand_builder.joint_dof_count
        self.num_hand_bodies = hand_builder.body_count

        # Set joint control parameters
        for i in range(self.num_hand_dofs):
            hand_builder.joint_target_ke[i] = self.config.hand_stiffness
            hand_builder.joint_target_kd[i] = self.config.hand_damping
            hand_builder.joint_target_pos[i] = 0.0
            hand_builder.joint_act_mode[i] = int(ActuatorMode.POSITION)

        # Store joint limits
        self.joint_lower = torch.tensor(
            hand_builder.joint_limit_lower[6:22], dtype=torch.float32, device=self.torch_device
        )
        self.joint_upper = torch.tensor(
            hand_builder.joint_limit_upper[6:22], dtype=torch.float32, device=self.torch_device
        )

        # Compute joint range for normalization
        self.joint_range = self.joint_upper - self.joint_lower
        self.joint_mid = (self.joint_upper + self.joint_lower) / 2

        # Replicate for all environments
        builder = newton.ModelBuilder()
        builder.replicate(hand_builder, self.num_envs, spacing=(0.5, 0.5, 0.0))

        builder.default_shape_cfg.ke = 1.0e4
        builder.default_shape_cfg.kd = 1.0e3
        builder.add_ground_plane()

        self.model = builder.finalize()

        # Solver
        self.solver = newton.solvers.SolverMuJoCo(
            self.model,
            solver="newton",
            integrator="implicitfast",
            njmax=300,
            nconmax=200,
            impratio=10.0,
            cone="elliptic",
            iterations=100,
            ls_iterations=50,
            use_mujoco_cpu=False,
        )

        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()
        self.contacts = self.model.collide(self.state_0)

        self.cube_body_offset = self.num_hand_bodies - 1
        self.hand_joint_offset = 6

    def _store_initial_state(self):
        """Store initial state for resetting environments."""
        self.initial_joint_q = wp.clone(self.state_0.joint_q)
        self.initial_joint_qd = wp.clone(self.state_0.joint_qd)
        newton.eval_fk(self.model, self.state_0.joint_q, self.state_0.joint_qd, self.state_0)

        # Initialize current targets to initial joint positions
        joint_q_np = self.initial_joint_q.numpy()
        for i in range(self.num_envs):
            start = i * self.num_hand_dofs + self.hand_joint_offset
            end = start + self.num_actions
            self.current_targets[i] = torch.from_numpy(joint_q_np[start:end]).to(self.torch_device)

    def _simulate_step(self):
        """Run one frame of simulation."""
        self.contacts = self.model.collide(self.state_0)

        for _ in range(self.sim_substeps):
            self.state_0.clear_forces()
            self.solver.step(self.state_0, self.state_1, self.control, self.contacts, self.sim_dt)
            self.state_0, self.state_1 = self.state_1, self.state_0

    def _random_quaternion(self, n: int) -> torch.Tensor:
        """Generate random unit quaternions (uniform on SO(3))."""
        u = torch.rand(n, 3, device=self.torch_device)
        q = torch.zeros(n, 4, device=self.torch_device)

        sqrt1_u0 = torch.sqrt(1 - u[:, 0])
        sqrt_u0 = torch.sqrt(u[:, 0])

        q[:, 0] = sqrt1_u0 * torch.sin(2 * np.pi * u[:, 1])  # x
        q[:, 1] = sqrt1_u0 * torch.cos(2 * np.pi * u[:, 1])  # y
        q[:, 2] = sqrt_u0 * torch.sin(2 * np.pi * u[:, 2])   # z
        q[:, 3] = sqrt_u0 * torch.cos(2 * np.pi * u[:, 2])   # w

        return q

    def reset(self, env_ids: torch.Tensor | None = None) -> torch.Tensor:
        """Reset specified environments."""
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.torch_device)

        num_reset = len(env_ids)
        if num_reset == 0:
            return self._compute_observations()

        # Reset episode counters
        self.episode_step[env_ids] = 0
        self.successes[env_ids] = 0
        self.consecutive_successes[env_ids] = 0

        # Reset joint states with small noise
        joint_q_np = self.state_0.joint_q.numpy()
        joint_qd_np = self.state_0.joint_qd.numpy()
        initial_q_np = self.initial_joint_q.numpy()
        initial_qd_np = self.initial_joint_qd.numpy()

        dofs_per_env = self.num_hand_dofs
        for idx in env_ids.cpu().numpy():
            start = idx * dofs_per_env
            end = start + dofs_per_env
            noise = np.random.uniform(-0.1, 0.1, size=dofs_per_env) * 0.1
            joint_q_np[start:end] = initial_q_np[start:end] + noise
            joint_qd_np[start:end] = initial_qd_np[start:end]

        self.state_0.joint_q = wp.array(joint_q_np, dtype=wp.float32, device=self.device)
        self.state_0.joint_qd = wp.array(joint_qd_np, dtype=wp.float32, device=self.device)

        # Recompute FK
        newton.eval_fk(self.model, self.state_0.joint_q, self.state_0.joint_qd, self.state_0)

        # Randomize goal quaternion (DextrEme style: full random rotation)
        if self.config.randomize_goal:
            self.goal_quat[env_ids] = self._random_quaternion(num_reset)
        else:
            # Simple z-axis rotation
            random_angles = torch.rand(num_reset, device=self.torch_device) * 2 * np.pi
            self.goal_quat[env_ids, 0] = 0.0
            self.goal_quat[env_ids, 1] = 0.0
            self.goal_quat[env_ids, 2] = torch.sin(random_angles / 2)
            self.goal_quat[env_ids, 3] = torch.cos(random_angles / 2)

        # Reset action buffers
        self.prev_actions[env_ids] = 0.0
        self.actions[env_ids] = 0.0

        # Reset current targets
        for idx in env_ids.cpu().numpy():
            start = idx * self.num_hand_dofs + self.hand_joint_offset
            end = start + self.num_actions
            self.current_targets[idx] = torch.from_numpy(joint_q_np[start:end]).to(self.torch_device)

        return self._compute_observations()

    def step(self, actions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict]:
        """Take a step in all environments."""
        # Store previous actions
        self.prev_actions = self.actions.clone()
        self.actions = actions.clone()

        # Apply delta actions with scaling
        if self.config.use_relative_control:
            action_delta = actions * self.config.action_scale
            self.current_targets = self.current_targets + action_delta
        else:
            # Absolute control
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

        # Compute rewards (DextrEme style)
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

        for i in range(self.num_envs):
            start = i * self.num_hand_dofs + self.hand_joint_offset
            end = start + self.num_actions
            control_np[start:end] = target_pos[i].cpu().numpy()

        self.control.joint_target_pos = wp.array(control_np, dtype=wp.float32, device=self.device)

    def _compute_observations(self) -> torch.Tensor:
        """Compute observations for all environments."""
        joint_q = torch.from_numpy(self.state_0.joint_q.numpy()).to(self.torch_device)
        joint_qd = torch.from_numpy(self.state_0.joint_qd.numpy()).to(self.torch_device)
        body_q = torch.from_numpy(self.state_0.body_q.numpy()).to(self.torch_device)
        body_qd = torch.from_numpy(self.state_0.body_qd.numpy()).to(self.torch_device)

        # Clamp velocities
        joint_qd = torch.clamp(joint_qd, -50.0, 50.0)

        for i in range(self.num_envs):
            q_start = i * self.num_hand_dofs + self.hand_joint_offset
            q_end = q_start + self.num_actions

            # Joint positions (normalized to [-1, 1])
            joint_pos = joint_q[q_start:q_end]
            self.obs_buf[i, 0:16] = (joint_pos - self.joint_mid) / (self.joint_range / 2 + 1e-6)

            # Joint velocities (scaled by 0.2)
            self.obs_buf[i, 16:32] = joint_qd[q_start:q_end] * 0.2

            # Cube state
            cube_idx = i * self.num_hand_bodies + self.cube_body_offset
            hand_idx = i * self.num_hand_bodies

            # Cube position relative to hand
            cube_pos = body_q[cube_idx, :3]
            hand_pos = body_q[hand_idx, :3]
            self.obs_buf[i, 32:35] = cube_pos - hand_pos

            # Cube orientation (x, y, z, w)
            self.obs_buf[i, 35:39] = body_q[cube_idx, 3:7]

            # Cube velocities (scaled)
            self.obs_buf[i, 39:42] = body_qd[cube_idx, :3] * 0.2
            self.obs_buf[i, 42:45] = body_qd[cube_idx, 3:6] * 0.2

            # Goal orientation
            self.obs_buf[i, 45:49] = self.goal_quat[i]

        # Handle NaN
        self.obs_buf = torch.nan_to_num(self.obs_buf, nan=0.0, posinf=5.0, neginf=-5.0)
        self.obs_buf = torch.clamp(self.obs_buf, -5.0, 5.0)

        return self.obs_buf

    def _quat_mul(self, q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
        """Multiply two quaternions (x, y, z, w format)."""
        x1, y1, z1, w1 = q1[..., 0], q1[..., 1], q1[..., 2], q1[..., 3]
        x2, y2, z2, w2 = q2[..., 0], q2[..., 1], q2[..., 2], q2[..., 3]

        w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
        x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
        y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
        z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2

        return torch.stack([x, y, z, w], dim=-1)

    def _quat_conjugate(self, q: torch.Tensor) -> torch.Tensor:
        """Compute quaternion conjugate."""
        return torch.stack([-q[..., 0], -q[..., 1], -q[..., 2], q[..., 3]], dim=-1)

    def _rotation_distance(self, q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
        """Compute rotation distance between quaternions (DextrEme style).

        Returns angle in radians: 2 * arcsin(||q_diff.xyz||)
        """
        # q_diff = q1 * conjugate(q2)
        q2_conj = self._quat_conjugate(q2)
        q_diff = self._quat_mul(q1, q2_conj)

        # Rotation distance = 2 * arcsin(clamp(||xyz||, max=1))
        xyz_norm = torch.norm(q_diff[..., :3], dim=-1)
        xyz_norm = torch.clamp(xyz_norm, max=1.0)
        return 2.0 * torch.asin(xyz_norm)

    def _compute_rewards(self) -> torch.Tensor:
        """Compute rewards (DextrEme style)."""
        body_q = torch.from_numpy(self.state_0.body_q.numpy()).to(self.torch_device)
        joint_qd = torch.from_numpy(self.state_0.joint_qd.numpy()).to(self.torch_device)

        for i in range(self.num_envs):
            cube_idx = i * self.num_hand_bodies + self.cube_body_offset
            hand_idx = i * self.num_hand_bodies

            # Get cube and hand states
            cube_quat = body_q[cube_idx, 3:7]  # x, y, z, w
            cube_pos = body_q[cube_idx, :3]
            hand_pos = body_q[hand_idx, :3]

            # 1. Rotation reward: 1 / (rot_dist + eps) * scale
            rot_dist = self._rotation_distance(cube_quat.unsqueeze(0), self.goal_quat[i].unsqueeze(0)).squeeze()
            rot_reward = self.config.rot_reward_scale / (rot_dist + self.config.rot_eps)

            # 2. Distance penalty (cube too far from hand)
            dist = torch.norm(cube_pos - hand_pos)
            dist_penalty = self.config.dist_reward_scale * dist

            # 3. Action penalty
            action_penalty = self.config.action_penalty_scale * torch.sum(self.actions[i] ** 2)

            # 4. Action delta penalty
            action_delta = self.actions[i] - self.prev_actions[i]
            action_delta_penalty = self.config.action_delta_penalty_scale * torch.sum(action_delta ** 2)

            # 5. Velocity penalty (DextrEme: -0.05 * sum((dof_vel/4)^2))
            q_start = i * self.num_hand_dofs + self.hand_joint_offset
            q_end = q_start + self.num_actions
            dof_vel = joint_qd[q_start:q_end]
            vel_penalty = self.config.velocity_penalty_scale * torch.sum((dof_vel / self.config.velocity_norm) ** 2)

            # 6. Success bonus (consecutive successes)
            success_bonus = 0.0
            if rot_dist < self.config.success_tolerance:
                self.consecutive_successes[i] += 1
                if self.consecutive_successes[i] >= self.config.consecutive_successes:
                    success_bonus = self.config.reach_goal_bonus
                    self.successes[i] += 1
                    # Generate new goal after success
                    if self.config.randomize_goal:
                        self.goal_quat[i] = self._random_quaternion(1).squeeze()
                    self.consecutive_successes[i] = 0
            else:
                self.consecutive_successes[i] = 0

            # 7. Fall penalty
            fall_penalty = 0.0
            if dist > self.config.fall_dist or cube_pos[2] < 0.05:
                fall_penalty = self.config.fall_penalty

            # Total reward
            self.reward_buf[i] = (
                rot_reward
                + dist_penalty
                - action_penalty
                - action_delta_penalty
                - vel_penalty
                + success_bonus
                + fall_penalty
            )

        return self.reward_buf

    def _compute_dones(self) -> torch.Tensor:
        """Check termination conditions."""
        body_q = torch.from_numpy(self.state_0.body_q.numpy()).to(self.torch_device)

        for i in range(self.num_envs):
            timeout = self.episode_step[i] >= self.max_episode_length

            cube_idx = i * self.num_hand_bodies + self.cube_body_offset
            hand_idx = i * self.num_hand_bodies

            cube_pos = body_q[cube_idx, :3]
            hand_pos = body_q[hand_idx, :3]
            dist = torch.norm(cube_pos - hand_pos)

            # Fail if cube falls or goes too far
            cube_fell = cube_pos[2] < 0.05
            cube_far = dist > self.config.fall_dist

            self.done_buf[i] = timeout or cube_fell or cube_far

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
