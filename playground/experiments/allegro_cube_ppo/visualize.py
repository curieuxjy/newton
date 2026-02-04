"""Visualize trained policy with Newton viewer."""

import argparse
import re

import numpy as np
import torch
import warp as wp

import newton
import newton.examples
from newton import ActuatorMode, ShapeFlags

from .config import EnvConfig
from .ppo import ActorCritic


class VisualizeExample:
    """Example class for visualizing trained Allegro Hand policy."""

    def __init__(self, viewer, checkpoint_path: str | None = None, num_envs: int = 4, show_collision: bool = True):
        self.fps = 60
        self.frame_dt = 1.0 / self.fps
        self.sim_substeps = 2
        self.sim_dt = self.frame_dt / self.sim_substeps
        self.control_decimation = 2
        self.action_scale = 0.3

        self.num_envs = num_envs
        self.viewer = viewer
        self.sim_time = 0.0
        self.show_collision = show_collision

        self.device = wp.get_device()
        self.torch_device = "cuda" if self.device.is_cuda else "cpu"

        # Build simulation
        self._build_simulation()

        # Compute joint range for action scaling
        self.joint_range = self.joint_upper - self.joint_lower
        self.joint_mid = (self.joint_upper + self.joint_lower) / 2

        # Load policy
        self.policy = None
        if checkpoint_path:
            self._load_policy(checkpoint_path)
            print(f"[INFO] Loaded policy from: {checkpoint_path}")
        else:
            print("[INFO] No policy loaded, using random actions")

        # Action buffer
        self.actions = torch.zeros(self.num_envs, 16, dtype=torch.float32, device=self.torch_device)

        # Current joint targets (for delta actions)
        self.current_targets = torch.zeros(self.num_envs, 16, dtype=torch.float32, device=self.torch_device)
        # Initialize to mid position
        self.current_targets[:] = self.joint_mid

        # Goal quaternion (for visualization)
        self.goal_quat = self._random_quaternion(self.num_envs)

        # Success tracking
        self.consecutive_successes = torch.zeros(self.num_envs, dtype=torch.int32, device=self.torch_device)
        self.success_tolerance = 0.4  # radians

        # Frame visualization parameters
        self.axis_length = 0.05  # Length of coordinate axes
        self.goal_offset_z = 0.1  # Goal frame offset above cube

        self.viewer.set_model(self.model)
        # Disable viewer's automatic world offsets since we already have physical spacing from replicate()
        # This ensures log_lines() coordinates match the rendered model positions
        self.viewer.set_world_offsets((0.0, 0.0, 0.0))

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

    def _build_simulation(self):
        """Build Newton simulation."""
        hand_builder = newton.ModelBuilder()
        newton.solvers.SolverMuJoCo.register_custom_attributes(hand_builder)

        hand_builder.default_shape_cfg.ke = 1.0e4
        hand_builder.default_shape_cfg.kd = 1.0e3
        hand_builder.default_shape_cfg.mu = 1.2

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

        # Show/hide collision shapes
        if not self.show_collision:
            # Hide collision shapes for cleaner visualization
            for i, key in enumerate(hand_builder.shape_key):
                if re.match(".*collision.*", key, re.IGNORECASE):
                    hand_builder.shape_flags[i] &= ~ShapeFlags.VISIBLE
        else:
            # Make sure collision shapes are visible
            for i, key in enumerate(hand_builder.shape_key):
                if re.match(".*collision.*", key, re.IGNORECASE):
                    hand_builder.shape_flags[i] |= ShapeFlags.VISIBLE
            print(f"[INFO] Collision shapes visible: {len(hand_builder.shape_key)} shapes")

        # Set joint control (lower stiffness for more compliant motion)
        for i in range(self.num_hand_dofs):
            hand_builder.joint_target_ke[i] = 40.0
            hand_builder.joint_target_kd[i] = 2.0
            hand_builder.joint_target_pos[i] = 0.0
            hand_builder.joint_act_mode[i] = int(ActuatorMode.POSITION)

        # Store joint limits
        self.joint_lower = torch.tensor(
            hand_builder.joint_limit_lower[6:22], dtype=torch.float32, device=self.torch_device
        )
        self.joint_upper = torch.tensor(
            hand_builder.joint_limit_upper[6:22], dtype=torch.float32, device=self.torch_device
        )

        # Replicate
        builder = newton.ModelBuilder()
        builder.replicate(hand_builder, self.num_envs, spacing=(0.5, 0.5, 0.0))

        builder.default_shape_cfg.ke = 1.0e3
        builder.default_shape_cfg.kd = 1.0e2
        builder.add_ground_plane()

        self.model = builder.finalize()

        self.solver = newton.solvers.SolverMuJoCo(
            self.model,
            solver="newton",
            integrator="implicitfast",
            njmax=200,
            nconmax=150,
            iterations=100,
            ls_iterations=50,
            use_mujoco_cpu=False,
        )

        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()
        self.contacts = self.model.collide(self.state_0)

        newton.eval_fk(self.model, self.state_0.joint_q, self.state_0.joint_qd, self.state_0)

    def _load_policy(self, checkpoint_path: str):
        """Load trained policy."""
        num_obs = 49  # Updated: 16 + 16 + 3 + 4 + 3 + 3 + 4
        num_actions = 16

        self.policy = ActorCritic(num_obs, num_actions, hidden_dims=(512, 256, 128))
        checkpoint = torch.load(checkpoint_path, map_location=self.torch_device)
        self.policy.load_state_dict(checkpoint["actor_critic"])
        self.policy.to(self.torch_device)
        self.policy.eval()

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
        """Compute rotation distance between quaternions."""
        q2_conj = self._quat_conjugate(q2)
        q_diff = self._quat_mul(q1, q2_conj)
        xyz_norm = torch.norm(q_diff[..., :3], dim=-1)
        xyz_norm = torch.clamp(xyz_norm, max=1.0)
        return 2.0 * torch.asin(xyz_norm)

    def _quat_rotate_vec(self, q: np.ndarray, v: np.ndarray) -> np.ndarray:
        """Rotate vector v by quaternion q (x, y, z, w format)."""
        # q = (x, y, z, w)
        qx, qy, qz, qw = q[0], q[1], q[2], q[3]

        # Rotation matrix from quaternion
        r00 = 1 - 2 * (qy * qy + qz * qz)
        r01 = 2 * (qx * qy - qz * qw)
        r02 = 2 * (qx * qz + qy * qw)

        r10 = 2 * (qx * qy + qz * qw)
        r11 = 1 - 2 * (qx * qx + qz * qz)
        r12 = 2 * (qy * qz - qx * qw)

        r20 = 2 * (qx * qz - qy * qw)
        r21 = 2 * (qy * qz + qx * qw)
        r22 = 1 - 2 * (qx * qx + qy * qy)

        return np.array([
            r00 * v[0] + r01 * v[1] + r02 * v[2],
            r10 * v[0] + r11 * v[1] + r12 * v[2],
            r20 * v[0] + r21 * v[1] + r22 * v[2],
        ])

    def _quat_mul_np(self, q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
        """Multiply two quaternions (x, y, z, w format) using numpy."""
        x1, y1, z1, w1 = q1[0], q1[1], q1[2], q1[3]
        x2, y2, z2, w2 = q2[0], q2[1], q2[2], q2[3]

        w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
        x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
        y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
        z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2

        return np.array([x, y, z, w])

    def _check_success_and_update_goals(self):
        """Check if cube orientation matches goal and update goals on success."""
        body_q = torch.from_numpy(self.state_0.body_q.numpy()).to(self.torch_device)
        cube_body_offset = self.num_hand_bodies - 1

        for i in range(self.num_envs):
            cube_idx = i * self.num_hand_bodies + cube_body_offset
            cube_quat = body_q[cube_idx, 3:7]  # x, y, z, w

            rot_dist = self._rotation_distance(cube_quat.unsqueeze(0), self.goal_quat[i].unsqueeze(0)).squeeze()

            if rot_dist < self.success_tolerance:
                self.consecutive_successes[i] += 1
                if self.consecutive_successes[i] >= 5:
                    # Success! Generate new goal
                    self.goal_quat[i] = self._random_quaternion(1).squeeze()
                    self.consecutive_successes[i] = 0
                    print(f"[ENV {i}] Goal reached! New goal generated.")
            else:
                self.consecutive_successes[i] = 0

    def _compute_obs(self) -> torch.Tensor:
        """Compute observations (matching env.py)."""
        joint_q = torch.from_numpy(self.state_0.joint_q.numpy()).to(self.torch_device)
        joint_qd = torch.from_numpy(self.state_0.joint_qd.numpy()).to(self.torch_device)
        body_q = torch.from_numpy(self.state_0.body_q.numpy()).to(self.torch_device)
        body_qd = torch.from_numpy(self.state_0.body_qd.numpy()).to(self.torch_device)

        joint_qd = torch.clamp(joint_qd, -50.0, 50.0)

        obs = torch.zeros(self.num_envs, 49, dtype=torch.float32, device=self.torch_device)

        cube_body_offset = self.num_hand_bodies - 1

        for i in range(self.num_envs):
            q_start = i * self.num_hand_dofs + 6
            q_end = q_start + 16

            # Joint positions (normalized)
            joint_pos = joint_q[q_start:q_end]
            obs[i, 0:16] = (joint_pos - self.joint_mid) / (self.joint_range / 2 + 1e-6)

            # Joint velocities (scaled)
            obs[i, 16:32] = joint_qd[q_start:q_end] * 0.2

            cube_idx = i * self.num_hand_bodies + cube_body_offset
            hand_idx = i * self.num_hand_bodies

            # Cube position relative to hand
            obs[i, 32:35] = body_q[cube_idx, :3] - body_q[hand_idx, :3]
            # Cube orientation
            obs[i, 35:39] = body_q[cube_idx, 3:7]
            # Cube velocities
            obs[i, 39:42] = body_qd[cube_idx, :3] * 0.2
            obs[i, 42:45] = body_qd[cube_idx, 3:6] * 0.2
            # Goal
            obs[i, 45:49] = self.goal_quat[i]

        obs = torch.nan_to_num(obs, nan=0.0, posinf=5.0, neginf=-5.0)
        return torch.clamp(obs, -5.0, 5.0)

    def _apply_actions(self, actions: torch.Tensor):
        """Apply actions to simulation."""
        scaled = actions * 0.5
        target_pos = torch.clamp(scaled, self.joint_lower, self.joint_upper)

        control_np = self.control.joint_target_pos.numpy()
        for i in range(self.num_envs):
            start = i * self.num_hand_dofs + 6
            end = start + 16
            control_np[start:end] = target_pos[i].cpu().numpy()

        self.control.joint_target_pos = wp.array(control_np, dtype=wp.float32, device=self.device)

    def step(self):
        """Step simulation."""
        # Get observation and action
        obs = self._compute_obs()

        if self.policy is not None:
            with torch.no_grad():
                self.actions, _, _, _ = self.policy.get_action_and_value(obs)
        else:
            # Random actions for testing (larger for visibility)
            self.actions = torch.randn_like(self.actions) * 0.5

        # Apply delta actions (matching env.py)
        action_delta = self.actions * self.action_scale
        self.current_targets = self.current_targets + action_delta
        self.current_targets = torch.clamp(self.current_targets, self.joint_lower, self.joint_upper)

        self._apply_actions(self.current_targets)

        # Step simulation
        for _ in range(self.control_decimation):
            self.contacts = self.model.collide(self.state_0)
            for _ in range(self.sim_substeps):
                self.state_0.clear_forces()
                self.viewer.apply_forces(self.state_0)
                self.solver.step(self.state_0, self.state_1, self.control, self.contacts, self.sim_dt)
                self.state_0, self.state_1 = self.state_1, self.state_0

        self.sim_time += self.frame_dt * self.control_decimation

        # Check for success and update goals
        self._check_success_and_update_goals()

    def _render_coordinate_frames(self):
        """Render coordinate frames for cube (current) and goal (target)."""
        body_q_np = self.state_0.body_q.numpy()
        cube_body_offset = self.num_hand_bodies - 1

        axis_length = self.axis_length

        # Unit vectors for axes
        x_axis = np.array([axis_length, 0.0, 0.0], dtype=np.float32)
        y_axis = np.array([0.0, axis_length, 0.0], dtype=np.float32)
        z_axis = np.array([0.0, 0.0, axis_length], dtype=np.float32)

        # Colors: X=Red, Y=Green, Z=Blue
        colors = np.array([
            [1.0, 0.0, 0.0],  # Red X
            [0.0, 1.0, 0.0],  # Green Y
            [0.0, 0.0, 1.0],  # Blue Z
        ], dtype=np.float32)

        for i in range(self.num_envs):
            cube_body_idx = i * self.num_hand_bodies + cube_body_offset
            cube_pos = body_q_np[cube_body_idx, :3].astype(np.float32)
            cube_quat = body_q_np[cube_body_idx, 3:7].astype(np.float32)

            # Rotate axes by cube orientation
            cube_x = self._quat_rotate_vec(cube_quat, x_axis)
            cube_y = self._quat_rotate_vec(cube_quat, y_axis)
            cube_z = self._quat_rotate_vec(cube_quat, z_axis)

            # Create arrays for this env's cube frame (3 lines)
            cube_starts = np.array([cube_pos, cube_pos, cube_pos], dtype=np.float32)
            cube_ends = np.array([
                cube_pos + cube_x,
                cube_pos + cube_y,
                cube_pos + cube_z,
            ], dtype=np.float32)

            # Log cube frame for this env
            self.viewer.log_lines(
                f"/cube_frame_{i}",
                wp.array(cube_starts, dtype=wp.vec3),
                wp.array(cube_ends, dtype=wp.vec3),
                wp.array(colors, dtype=wp.vec3),
                width=0.003,
            )

            # Goal frame (above cube)
            goal_pos = cube_pos.copy()
            goal_pos[2] += self.goal_offset_z
            goal_quat = self.goal_quat[i].cpu().numpy().astype(np.float32)

            goal_x = self._quat_rotate_vec(goal_quat, x_axis)
            goal_y = self._quat_rotate_vec(goal_quat, y_axis)
            goal_z = self._quat_rotate_vec(goal_quat, z_axis)

            goal_starts = np.array([goal_pos, goal_pos, goal_pos], dtype=np.float32)
            goal_ends = np.array([
                goal_pos + goal_x,
                goal_pos + goal_y,
                goal_pos + goal_z,
            ], dtype=np.float32)

            # Log goal frame for this env
            self.viewer.log_lines(
                f"/goal_frame_{i}",
                wp.array(goal_starts, dtype=wp.vec3),
                wp.array(goal_ends, dtype=wp.vec3),
                wp.array(colors, dtype=wp.vec3),
                width=0.003,
            )

    def render(self):
        """Render frame."""
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        self.viewer.log_contacts(self.contacts, self.state_0)

        # Render coordinate frames for cube and goal
        self._render_coordinate_frames()

        self.viewer.end_frame()

    def test_final(self):
        """Required for Newton examples."""
        pass


def main():
    parser = newton.examples.create_parser()
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to trained checkpoint")
    parser.add_argument("--num-envs", type=int, default=4, help="Number of environments to visualize")
    parser.add_argument("--show-collision", action="store_true", default=True, help="Show collision shapes")
    parser.add_argument("--hide-collision", action="store_false", dest="show_collision", help="Hide collision shapes")

    viewer, args = newton.examples.init(parser)

    example = VisualizeExample(
        viewer,
        checkpoint_path=args.checkpoint,
        num_envs=args.num_envs,
        show_collision=args.show_collision,
    )

    newton.examples.run(example, args)


if __name__ == "__main__":
    main()
