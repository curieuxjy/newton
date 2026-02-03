"""Allegro Hand Cube Rotation Environment for RL training with Newton."""

from typing import Any

import numpy as np
import torch
import warp as wp

import newton
from newton import ActuatorMode

from .config import EnvConfig


class AllegroHandCubeEnv:
    """Vectorized environment for Allegro Hand cube rotation task.

    This environment simulates multiple parallel Allegro hands manipulating cubes,
    designed for efficient RL training with PPO.

    Observation space (46 dims):
        - Hand joint positions (16)
        - Hand joint velocities (16)
        - Cube position relative to palm (3)
        - Cube orientation quaternion (4)
        - Goal rotation quaternion (4)
        - Previous action (16) - optional, for action smoothness

    Action space (16 dims):
        - Joint position targets for 16 actuated DOFs
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
        self.num_obs = 16 + 16 + 3 + 4 + 4  # joint pos + vel + cube pos + cube quat + goal quat
        self.num_actions = 16  # 16 actuated DOFs

        # Buffers
        self.obs_buf = torch.zeros(self.num_envs, self.num_obs, dtype=torch.float32, device=self.torch_device)
        self.reward_buf = torch.zeros(self.num_envs, dtype=torch.float32, device=self.torch_device)
        self.done_buf = torch.zeros(self.num_envs, dtype=torch.bool, device=self.torch_device)
        self.info_buf: dict[str, Any] = {}

        # Action buffers
        self.actions = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float32, device=self.torch_device)
        self.prev_actions = torch.zeros_like(self.actions)

        # Goal state
        self.goal_quat = torch.zeros(self.num_envs, 4, dtype=torch.float32, device=self.torch_device)
        self.goal_quat[:, 3] = 1.0  # identity quaternion (x, y, z, w)
        self.goal_rotation_speed = config.goal_rotation_speed

        # Initial state storage for reset
        self._store_initial_state()

        # CUDA graph for simulation (disabled for now, enable after debugging)
        self.graph = None
        # if self.device.is_cuda:
        #     self._capture_cuda_graph()

    def _build_simulation(self):
        """Build the Newton simulation with Allegro hand and cube."""
        # Build single hand+cube model
        hand_builder = newton.ModelBuilder()
        newton.solvers.SolverMuJoCo.register_custom_attributes(hand_builder)

        # Contact parameters
        hand_builder.default_shape_cfg.ke = 1.0e3
        hand_builder.default_shape_cfg.kd = 1.0e2
        hand_builder.default_shape_cfg.mu = 1.0

        # Load Allegro hand from downloaded asset
        asset_path = newton.utils.download_asset("wonik_allegro")
        # Note: Using left hand as the asset includes cube; can switch to right if available
        asset_file = str(asset_path / "usd" / "allegro_left_hand_with_cube.usda")

        hand_builder.add_usd(
            asset_file,
            xform=wp.transform(wp.vec3(0, 0, 0.3)),
            ignore_paths=[".*Dummy", ".*CollisionPlane", ".*goal", ".*DexCube/visuals"],
        )

        # Count DOFs (should be 22: 6 for root + 16 for fingers)
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

        # Replicate for all environments
        builder = newton.ModelBuilder()
        builder.replicate(hand_builder, self.num_envs, spacing=(0.5, 0.5, 0.0))

        # Add ground plane
        builder.default_shape_cfg.ke = 1.0e3
        builder.default_shape_cfg.kd = 1.0e2
        builder.add_ground_plane()

        # Finalize model
        self.model = builder.finalize()

        # Create solver
        self.solver = newton.solvers.SolverMuJoCo(
            self.model,
            solver="newton",
            integrator="implicitfast",
            njmax=200,
            nconmax=150,
            impratio=10.0,
            cone="elliptic",
            iterations=100,
            ls_iterations=50,
            use_mujoco_cpu=False,
        )

        # Create states and control
        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()
        self.contacts = self.model.collide(self.state_0)

        # Compute body/joint indices for each environment
        self.cube_body_offset = self.num_hand_bodies - 1  # cube is last body in each hand model
        self.hand_joint_offset = 6  # skip 6 DOFs for root joint

    def _store_initial_state(self):
        """Store initial state for resetting environments."""
        self.initial_joint_q = wp.clone(self.state_0.joint_q)
        self.initial_joint_qd = wp.clone(self.state_0.joint_qd)
        newton.eval_fk(self.model, self.state_0.joint_q, self.state_0.joint_qd, self.state_0)
        self.initial_body_q = wp.clone(self.state_0.body_q)

    def _capture_cuda_graph(self):
        """Capture CUDA graph for fast simulation stepping."""
        if wp.is_mempool_enabled(self.device):
            with wp.ScopedCapture() as capture:
                self._simulate_step()
            self.graph = capture.graph

    def _simulate_step(self):
        """Run one frame of simulation (multiple substeps)."""
        self.contacts = self.model.collide(self.state_0)

        for _ in range(self.sim_substeps):
            self.state_0.clear_forces()
            self.solver.step(self.state_0, self.state_1, self.control, self.contacts, self.sim_dt)
            self.state_0, self.state_1 = self.state_1, self.state_0

    def reset(self, env_ids: torch.Tensor | None = None) -> torch.Tensor:
        """Reset specified environments or all if env_ids is None."""
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.torch_device)

        num_reset = len(env_ids)
        if num_reset == 0:
            return self._compute_observations()

        # Reset episode counters
        self.episode_step[env_ids] = 0

        # Reset joint states for selected environments
        # This is a simplified reset - for production, use Warp kernels
        joint_q_np = self.state_0.joint_q.numpy()
        joint_qd_np = self.state_0.joint_qd.numpy()
        initial_q_np = self.initial_joint_q.numpy()
        initial_qd_np = self.initial_joint_qd.numpy()

        dofs_per_env = self.num_hand_dofs
        for idx in env_ids.cpu().numpy():
            start = idx * dofs_per_env
            end = start + dofs_per_env
            joint_q_np[start:end] = initial_q_np[start:end]
            joint_qd_np[start:end] = initial_qd_np[start:end]

        # Copy back to GPU
        self.state_0.joint_q = wp.array(joint_q_np, dtype=wp.float32, device=self.device)
        self.state_0.joint_qd = wp.array(joint_qd_np, dtype=wp.float32, device=self.device)

        # Recompute forward kinematics
        newton.eval_fk(self.model, self.state_0.joint_q, self.state_0.joint_qd, self.state_0)

        # Reset goal quaternions (random rotation around z-axis)
        random_angles = torch.rand(num_reset, device=self.torch_device) * 2 * np.pi
        self.goal_quat[env_ids, 0] = 0.0
        self.goal_quat[env_ids, 1] = 0.0
        self.goal_quat[env_ids, 2] = torch.sin(random_angles / 2)
        self.goal_quat[env_ids, 3] = torch.cos(random_angles / 2)

        # Reset action buffers
        self.prev_actions[env_ids] = 0.0

        # Recompute observations
        return self._compute_observations()

    def step(self, actions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict]:
        """Take a step in all environments.

        Args:
            actions: Joint position targets, shape (num_envs, 16)

        Returns:
            observations, rewards, dones, info
        """
        # Store previous actions for rate penalty
        self.prev_actions = self.actions.clone()
        self.actions = actions.clone()

        # Scale and clamp actions to joint limits
        scaled_actions = actions * 0.5  # action scale
        target_pos = torch.clamp(scaled_actions, self.joint_lower, self.joint_upper)

        # Apply actions to control (need to offset by root DOFs)
        self._apply_actions(target_pos)

        # Step simulation (with control decimation)
        for _ in range(self.control_decimation):
            if self.graph:
                wp.capture_launch(self.graph)
            else:
                self._simulate_step()

        # Update episode step
        self.episode_step += 1

        # Compute observations
        obs = self._compute_observations()

        # Compute rewards
        rewards = self._compute_rewards()

        # Check termination
        dones = self._compute_dones()

        # Update goal (rotating target)
        self._update_goal()

        # Auto-reset done environments
        done_env_ids = torch.where(dones)[0]
        if len(done_env_ids) > 0:
            self.reset(done_env_ids)

        return obs, rewards, dones, self.info_buf

    def _apply_actions(self, target_pos: torch.Tensor):
        """Apply joint position targets to control buffer."""
        # Convert to Warp array and copy to control
        control_np = self.control.joint_target_pos.numpy()

        # Each environment has num_hand_dofs DOFs, we control DOFs 6:22 (fingers)
        for i in range(self.num_envs):
            start = i * self.num_hand_dofs + self.hand_joint_offset
            end = start + self.num_actions
            control_np[start:end] = target_pos[i].cpu().numpy()

        self.control.joint_target_pos = wp.array(control_np, dtype=wp.float32, device=self.device)

    def _compute_observations(self) -> torch.Tensor:
        """Compute observations for all environments."""
        # Get joint positions and velocities
        joint_q = torch.from_numpy(self.state_0.joint_q.numpy()).to(self.torch_device)
        joint_qd = torch.from_numpy(self.state_0.joint_qd.numpy()).to(self.torch_device)
        body_q = torch.from_numpy(self.state_0.body_q.numpy()).to(self.torch_device)

        # Clip velocities to prevent explosion
        joint_qd = torch.clamp(joint_qd, -100.0, 100.0)

        for i in range(self.num_envs):
            # Joint positions (16 finger DOFs)
            q_start = i * self.num_hand_dofs + self.hand_joint_offset
            q_end = q_start + self.num_actions
            self.obs_buf[i, 0:16] = joint_q[q_start:q_end]

            # Joint velocities (16 finger DOFs) - scaled down
            self.obs_buf[i, 16:32] = joint_qd[q_start:q_end] * 0.1

            # Cube position relative to hand root (3)
            cube_body_idx = i * self.num_hand_bodies + self.cube_body_offset
            hand_root_idx = i * self.num_hand_bodies
            cube_pos = body_q[cube_body_idx, :3]
            hand_pos = body_q[hand_root_idx, :3]
            self.obs_buf[i, 32:35] = cube_pos - hand_pos

            # Cube orientation (4)
            self.obs_buf[i, 35:39] = body_q[cube_body_idx, 3:7]

            # Goal orientation (4)
            self.obs_buf[i, 39:43] = self.goal_quat[i]

        # Replace NaN with zeros
        self.obs_buf = torch.nan_to_num(self.obs_buf, nan=0.0, posinf=10.0, neginf=-10.0)

        return self.obs_buf

    def _compute_rewards(self) -> torch.Tensor:
        """Compute rewards for all environments."""
        body_q = torch.from_numpy(self.state_0.body_q.numpy()).to(self.torch_device)

        for i in range(self.num_envs):
            cube_body_idx = i * self.num_hand_bodies + self.cube_body_offset
            cube_quat = body_q[cube_body_idx, 3:7]  # x, y, z, w

            # Rotation alignment reward (quaternion distance)
            goal_quat = self.goal_quat[i]
            quat_diff = self._quat_distance(cube_quat, goal_quat)
            rotation_reward = 1.0 - quat_diff

            # Action penalty
            action_penalty = torch.sum(self.actions[i] ** 2)

            # Action rate penalty
            action_rate = torch.sum((self.actions[i] - self.prev_actions[i]) ** 2)

            # Total reward
            self.reward_buf[i] = (
                self.config.reward_rotation * rotation_reward
                - self.config.reward_action_penalty * action_penalty
                - self.config.reward_action_rate_penalty * action_rate
            )

        return self.reward_buf

    def _quat_distance(self, q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
        """Compute distance between two quaternions (0 = identical, 1 = opposite)."""
        dot = torch.abs(torch.sum(q1 * q2))
        return 1.0 - dot

    def _compute_dones(self) -> torch.Tensor:
        """Check termination conditions."""
        body_q = torch.from_numpy(self.state_0.body_q.numpy()).to(self.torch_device)

        for i in range(self.num_envs):
            # Check if episode length exceeded
            timeout = self.episode_step[i] >= self.max_episode_length

            # Check if cube fell (z < 0.05)
            cube_body_idx = i * self.num_hand_bodies + self.cube_body_offset
            cube_z = body_q[cube_body_idx, 2]
            cube_fell = cube_z < 0.05

            self.done_buf[i] = timeout or cube_fell

        return self.done_buf

    def _update_goal(self):
        """Update goal quaternion (rotating target)."""
        # Increment goal rotation around z-axis
        angle_delta = self.goal_rotation_speed * self.frame_dt * self.control_decimation

        # Current angle from quaternion
        current_angle = 2 * torch.atan2(self.goal_quat[:, 2], self.goal_quat[:, 3])
        new_angle = current_angle + angle_delta

        # Update quaternion
        self.goal_quat[:, 0] = 0.0
        self.goal_quat[:, 1] = 0.0
        self.goal_quat[:, 2] = torch.sin(new_angle / 2)
        self.goal_quat[:, 3] = torch.cos(new_angle / 2)

    def close(self):
        """Clean up resources."""
        pass

    @property
    def observation_space(self):
        """Gymnasium-compatible observation space."""
        import gymnasium as gym

        return gym.spaces.Box(low=-np.inf, high=np.inf, shape=(self.num_obs,), dtype=np.float32)

    @property
    def action_space(self):
        """Gymnasium-compatible action space."""
        import gymnasium as gym

        return gym.spaces.Box(low=-1.0, high=1.0, shape=(self.num_actions,), dtype=np.float32)
