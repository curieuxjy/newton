"""Visualize trained policy with Newton viewer."""

import argparse

import torch
import warp as wp

import newton
import newton.examples
from newton import ActuatorMode

from .config import EnvConfig
from .ppo import ActorCritic


class VisualizeExample:
    """Example class for visualizing trained Allegro Hand policy."""

    def __init__(self, viewer, checkpoint_path: str | None = None, num_envs: int = 4):
        self.fps = 60
        self.frame_dt = 1.0 / self.fps
        self.sim_substeps = 2
        self.sim_dt = self.frame_dt / self.sim_substeps
        self.control_decimation = 2
        self.action_scale = 0.3

        self.num_envs = num_envs
        self.viewer = viewer
        self.sim_time = 0.0

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
        self.goal_quat = torch.zeros(self.num_envs, 4, dtype=torch.float32, device=self.torch_device)
        self.goal_quat[:, 3] = 1.0

        self.viewer.set_model(self.model)

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
            ignore_paths=[".*Dummy", ".*CollisionPlane", ".*goal", ".*DexCube/visuals"],
        )

        self.num_hand_dofs = hand_builder.joint_dof_count
        self.num_hand_bodies = hand_builder.body_count

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

    def render(self):
        """Render frame."""
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        self.viewer.log_contacts(self.contacts, self.state_0)
        self.viewer.end_frame()

    def test_final(self):
        """Required for Newton examples."""
        pass


def main():
    parser = newton.examples.create_parser()
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to trained checkpoint")
    parser.add_argument("--num-envs", type=int, default=4, help="Number of environments to visualize")

    viewer, args = newton.examples.init(parser)

    example = VisualizeExample(viewer, checkpoint_path=args.checkpoint, num_envs=args.num_envs)

    newton.examples.run(example, args)


if __name__ == "__main__":
    main()
