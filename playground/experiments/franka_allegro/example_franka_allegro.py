"""Example combining Franka arm with Allegro hand.

This example demonstrates how to attach an Allegro dexterous hand to a Franka arm's
end effector, creating a combined manipulation system.
"""

import numpy as np
import warp as wp

import newton
import newton.examples
from newton import ActuatorMode


class Example:
    """Franka arm with Allegro hand attached to the end effector."""

    def __init__(self, viewer, num_worlds: int = 1):
        self.fps = 60
        self.frame_dt = 1.0 / self.fps
        self.sim_substeps = 10
        self.sim_dt = self.frame_dt / self.sim_substeps

        self.sim_time = 0.0
        self.num_worlds = num_worlds
        self.viewer = viewer
        self.device = wp.get_device()

        # Build the combined robot
        self._build_robot()

        # Setup MuJoCo solver (following panda_hydro example)
        self.solver = newton.solvers.SolverMuJoCo(
            self.model,
            use_mujoco_contacts=False,
            solver="newton",
            integrator="implicitfast",
            cone="elliptic",
            njmax=15000,
            nconmax=15000,
            iterations=15,
            ls_iterations=100,
            ls_parallel=True,
            impratio=1000.0,
        )

        # Create state and control
        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()
        self.contacts = self.model.collide(self.state_0)

        # Initialize FK
        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.state_0)

        self.viewer.set_model(self.model)

        # Setup control parameters
        self._setup_control()

        self.capture()

    def _build_robot(self):
        """Build the combined Franka + Allegro robot."""
        builder = newton.ModelBuilder()

        # Load Franka arm (following panda_hydro example)
        franka_asset = newton.utils.download_asset("franka_emika_panda")
        builder.add_urdf(
            franka_asset / "urdf/fr3.urdf",  # Arm only (no fingers)
            xform=wp.transform((0.0, 0.0, 0.0), wp.quat_identity()),  # Base at origin
            enable_self_collisions=False,
        )

        # Store Franka info
        self.franka_body_count = builder.body_count
        self.franka_joint_count = builder.joint_count
        self.franka_dof_count = builder.joint_dof_count

        # Set Franka joint parameters (following panda_hydro)
        builder.joint_target_ke[:self.franka_dof_count] = [500.0] * self.franka_dof_count
        builder.joint_target_kd[:self.franka_dof_count] = [100.0] * self.franka_dof_count
        builder.joint_effort_limit[:self.franka_dof_count] = [80.0] * self.franka_dof_count
        builder.joint_armature[:self.franka_dof_count] = [0.1] * self.franka_dof_count

        # Set initial Franka pose to midpoints of joint limits
        franka_lower = np.array(builder.joint_limit_lower[:self.franka_dof_count])
        franka_upper = np.array(builder.joint_limit_upper[:self.franka_dof_count])
        init_franka_q = (franka_lower + franka_upper) / 2
        builder.joint_q[:self.franka_dof_count] = init_franka_q.tolist()
        builder.joint_target_pos[:self.franka_dof_count] = init_franka_q.tolist()
        print(f"[INFO] Franka initial pose (midpoints): {init_franka_q}")

        # Find end effector body (fr3_link8)
        ee_body_idx = -1
        for i, key in enumerate(builder.body_key):
            if "fr3_link8" in key:
                ee_body_idx = i
                print(f"[INFO] Found end effector body: {key} (index {i})")
                break

        if ee_body_idx < 0:
            ee_body_idx = self.franka_body_count - 1
            print(f"[INFO] Using fallback end effector body index: {ee_body_idx}")

        self.ee_body_idx = ee_body_idx

        # Load Allegro hand
        allegro_asset = newton.utils.download_asset("wonik_allegro")
        allegro_urdf = allegro_asset / "urdf/allegro_hand_description_left.urdf"

        body_offset = builder.body_count
        joint_offset = builder.joint_count
        dof_offset = builder.joint_dof_count

        # Transform for Allegro relative to Franka's end effector
        allegro_xform = wp.transform(
            wp.vec3(0.0, 0.0, 0.1),  # Offset from EE
            wp.quat_from_axis_angle(wp.vec3(0, 0, 1), np.pi),  # Rotate 180° around Z
        )

        builder.add_urdf(
            allegro_urdf,
            floating=False,
            xform=allegro_xform,
            enable_self_collisions=False,
        )

        self.allegro_body_offset = body_offset
        self.allegro_body_count = builder.body_count - body_offset
        self.allegro_joint_count = builder.joint_count - joint_offset
        self.allegro_dof_count = builder.joint_dof_count - dof_offset

        print(f"[INFO] Franka: {self.franka_body_count} bodies, {self.franka_dof_count} DOFs")
        print(f"[INFO] Allegro: {self.allegro_body_count} bodies, {self.allegro_dof_count} DOFs")

        # Connect Allegro root joint to Franka EE
        allegro_root_joint_idx = joint_offset
        if builder.joint_parent[allegro_root_joint_idx] == -1:
            builder.joint_parent[allegro_root_joint_idx] = ee_body_idx
            builder.joint_X_p[allegro_root_joint_idx] = allegro_xform
            print(f"[INFO] Connected Allegro to Franka EE (body {ee_body_idx})")

        # Merge articulations
        for j in range(joint_offset, builder.joint_count):
            builder.joint_articulation[j] = 0
        builder.articulation_start = [0]
        builder.articulation_key = ["franka_allegro"]
        builder.articulation_world = [0]

        # Set Allegro joint parameters (higher stiffness for stability)
        allegro_dof_start = dof_offset
        for i in range(allegro_dof_start, builder.joint_dof_count):
            builder.joint_target_ke[i] = 200.0  # Increased from 50
            builder.joint_target_kd[i] = 20.0   # Increased from 5
            builder.joint_effort_limit[i] = 20.0
            builder.joint_armature[i] = 0.05    # Increased from 0.01
            builder.joint_act_mode[i] = int(ActuatorMode.POSITION)

        # Set initial Allegro pose to midpoints of joint limits
        allegro_lower = np.array(builder.joint_limit_lower[allegro_dof_start:allegro_dof_start + self.allegro_dof_count])
        allegro_upper = np.array(builder.joint_limit_upper[allegro_dof_start:allegro_dof_start + self.allegro_dof_count])
        init_allegro_q = (allegro_lower + allegro_upper) / 2
        builder.joint_q[allegro_dof_start:allegro_dof_start + self.allegro_dof_count] = init_allegro_q.tolist()
        builder.joint_target_pos[allegro_dof_start:allegro_dof_start + self.allegro_dof_count] = init_allegro_q.tolist()
        print(f"[INFO] Allegro initial pose (midpoints): {init_allegro_q}")

        self.allegro_dof_start = allegro_dof_start

        # Store joint limits for animation (use actual URDF limits)
        self.franka_joint_lower = np.array(builder.joint_limit_lower[:self.franka_dof_count], dtype=np.float32)
        self.franka_joint_upper = np.array(builder.joint_limit_upper[:self.franka_dof_count], dtype=np.float32)
        self.allegro_joint_lower = np.array(builder.joint_limit_lower[allegro_dof_start:allegro_dof_start + self.allegro_dof_count], dtype=np.float32)
        self.allegro_joint_upper = np.array(builder.joint_limit_upper[allegro_dof_start:allegro_dof_start + self.allegro_dof_count], dtype=np.float32)

        print(f"[INFO] Franka joint limits: {self.franka_joint_lower} to {self.franka_joint_upper}")
        print(f"[INFO] Allegro joint limits:")
        finger_names = ["Ring", "Middle", "Index", "Thumb"]
        for f in range(4):
            start = f * 4
            end = start + 4
            print(f"  {finger_names[f]}: lower={self.allegro_joint_lower[start:end]}, upper={self.allegro_joint_upper[start:end]}")

        # Add ground plane
        builder.add_ground_plane()

        self.model = builder.finalize()

        print(f"[INFO] Total: {self.model.body_count} bodies, {self.model.joint_count} joints, {self.model.joint_dof_count} DOFs")

    def _setup_control(self):
        """Setup initial control targets."""
        # Compute joint midpoints and ranges
        self.franka_mid = (self.franka_joint_lower + self.franka_joint_upper) / 2
        self.franka_range = (self.franka_joint_upper - self.franka_joint_lower) / 2
        self.allegro_mid = (self.allegro_joint_lower + self.allegro_joint_upper) / 2
        self.allegro_range = (self.allegro_joint_upper - self.allegro_joint_lower) / 2

        # Different oscillation speeds for each joint (faster for visible motion)
        self.franka_speeds = np.array([1.0, 0.8, 1.2, 0.7, 1.5, 1.0, 1.8], dtype=np.float32)
        self.allegro_speeds = np.linspace(2.0, 4.0, self.allegro_dof_count, dtype=np.float32)

        # Phase offsets for more interesting motion
        self.franka_phases = np.linspace(0, 2 * np.pi, self.franka_dof_count, dtype=np.float32)
        self.allegro_phases = np.linspace(0, 4 * np.pi, self.allegro_dof_count, dtype=np.float32)

        self.anim_time = 0.0

    def capture(self):
        """Capture CUDA graph for simulation."""
        # Disable CUDA graph capture because control targets change every frame
        self.graph = None

    def simulate(self):
        """Run simulation step."""
        self.state_0.clear_forces()
        self.state_1.clear_forces()

        for i in range(self.sim_substeps):
            if i % 2 == 0:  # Collide every 2 substeps
                self.contacts = self.model.collide(self.state_0)
            self.solver.step(self.state_0, self.state_1, self.control, self.contacts, self.sim_dt)
            self.state_0, self.state_1 = self.state_1, self.state_0

    def step(self):
        """Step the simulation."""
        self.anim_time += self.frame_dt

        # Franka arm - keep at midpoint (no movement)
        franka_target = self.franka_mid.copy()

        # DOF mapping (Newton loads in tree order):
        # DOF 0-3: Ring finger (link_8-11), DOF 4-7: Middle finger (link_4-7)
        # DOF 8-11: Index finger (link_0-3), DOF 12-15: Thumb (link_12-15)
        # Each finger has 4 joints: Abduction, MCP, PIP, DIP
        finger_names = ["Ring", "Middle", "Index", "Thumb"]

        # Animate all 3 fingers (Ring, Middle, Index) simultaneously for comparison
        # Use same oscillation parameters for all to compare motion
        allegro_target = self.allegro_mid.copy()
        range_scale = 0.9

        # Animate fingers 0, 1, 2 (Ring, Middle, Index) with same motion
        for finger_idx in range(3):  # Ring, Middle, Index (not thumb)
            start_dof = finger_idx * 4
            for j in range(4):  # 4 joints per finger
                dof = start_dof + j
                t = np.sin(self.anim_time * 3.0)  # Same phase for comparison
                allegro_target[dof] = self.allegro_mid[dof] + t * self.allegro_range[dof] * range_scale

        # Apply control targets
        control_np = self.control.joint_target_pos.numpy()
        control_np[:self.franka_dof_count] = franka_target
        control_np[self.allegro_dof_start:self.allegro_dof_start + self.allegro_dof_count] = allegro_target
        self.control.joint_target_pos = wp.array(control_np, dtype=wp.float32, device=self.device)

        # Run simulation
        if self.graph:
            wp.capture_launch(self.graph)
        else:
            self.simulate()

        # Print status every 60 frames
        self.frame_count = getattr(self, 'frame_count', 0) + 1
        if self.frame_count % 60 == 0:
            print(f"[t={self.anim_time:.1f}s] All 3 fingers (Ring, Middle, Index) moving with same motion")

        self.sim_time += self.frame_dt

    def render(self):
        """Render the scene."""
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        self.viewer.log_contacts(self.contacts, self.state_0)
        self.viewer.end_frame()

    def test_final(self):
        """Validation test."""
        pass


if __name__ == "__main__":
    parser = newton.examples.create_parser()
    parser.add_argument("--num-worlds", type=int, default=1, help="Number of worlds")

    viewer, args = newton.examples.init(parser)
    example = Example(viewer, args.num_worlds)
    newton.examples.run(example, args)
