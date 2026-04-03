"""Isaac-Dexsuite-Kuka-Allegro-Lift-v0 implemented with Newton physics.

This example demonstrates a Kuka iiwa 14 arm with an Allegro dexterous hand
performing a cube lift task on a table. Replicates the IsaacLab environment
Isaac-Dexsuite-Kuka-Allegro-Lift-v0 using Newton's native simulation.

Usage:
    uv run --extra examples python playground/experiments/kuka_allegro_lift/example_kuka_allegro_lift.py
    uv run --extra examples python playground/experiments/kuka_allegro_lift/example_kuka_allegro_lift.py --world-count 4
"""

import math

import numpy as np
import warp as wp

import newton
import newton.examples
from newton import JointTargetMode
from newton._src.utils import compute_world_offsets
from newton._src.utils.download_assets import download_git_folder

# MuJoCo menagerie URL for Kuka
_MENAGERIE_URL = "https://github.com/google-deepmind/mujoco_menagerie.git"


class Example:
    """Kuka iiwa 14 + Allegro hand cube lift task.

    The robot reaches toward a cube on a table, grasps it, and lifts it.
    Uses a scripted policy that cycles through reach -> grasp -> lift phases.
    """

    def __init__(self, viewer, num_worlds: int = 1):
        self.fps = 60
        self.frame_dt = 1.0 / self.fps
        self.sim_substeps = 10
        self.sim_dt = self.frame_dt / self.sim_substeps

        self.sim_time = 0.0
        self.world_count = num_worlds
        self.viewer = viewer
        self.device = wp.get_device()

        self._build_scene()

        self.solver = newton.solvers.SolverMuJoCo(
            self.model,
            solver="newton",
            integrator="implicitfast",
            njmax=15000 * self.world_count,
            nconmax=15000 * self.world_count,
            impratio=1000.0,
        )

        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()
        self.contacts = self.model.collide(self.state_0)

        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.state_0)
        self.viewer.set_model(self.model)

        self._setup_control()
        self.graph = None  # No CUDA graph (dynamic control)

    def _build_scene(self):
        """Build Kuka + Allegro + table + cube scene with replication."""
        # === Build single world ===
        world_builder = newton.ModelBuilder()
        newton.solvers.SolverMuJoCo.register_custom_attributes(world_builder)

        world_builder.default_shape_cfg.ke = 2.5e3
        world_builder.default_shape_cfg.kd = 250.0

        # === Kuka iiwa 14 from MuJoCo menagerie ===
        kuka_path = download_git_folder(_MENAGERIE_URL, "kuka_iiwa_14")
        world_builder.add_mjcf(
            str(kuka_path / "iiwa14.xml"),
            xform=wp.transform(wp.vec3(0.0, 0.0, 0.0), wp.quat_identity()),
        )

        self.kuka_body_count = world_builder.body_count
        self.kuka_joint_count = world_builder.joint_count
        self.kuka_dof_count = world_builder.joint_dof_count

        # MJCF loader doesn't set position control by default - set explicitly.
        for i in range(self.kuka_dof_count):
            world_builder.joint_target_mode[i] = int(JointTargetMode.POSITION)
            world_builder.joint_target_ke[i] = 100000.0
            world_builder.joint_target_kd[i] = 10000.0
            world_builder.joint_effort_limit[i] = 5000.0
            world_builder.joint_armature[i] = 0.1

        # Initial pose: joint0 = -90° to face -Y
        kuka_init_q = [-math.pi / 2, 0.4, 0.0, -1.4, 0.0, 1.2, 0.0]
        world_builder.joint_q[:self.kuka_dof_count] = kuka_init_q
        world_builder.joint_target_pos[:self.kuka_dof_count] = kuka_init_q

        # Find EE body (link7 = last body of Kuka)
        self.ee_body_idx = -1
        for i, key in enumerate(world_builder.body_label):
            if "link7" in key:
                self.ee_body_idx = i
                break
        if self.ee_body_idx < 0:
            self.ee_body_idx = self.kuka_body_count - 1

        # === Allegro hand ===
        allegro_asset = newton.utils.download_asset("wonik_allegro")
        allegro_urdf = allegro_asset / "urdf/allegro_hand_description_left.urdf"

        body_offset = world_builder.body_count
        joint_offset = world_builder.joint_count
        dof_offset = world_builder.joint_dof_count

        allegro_xform = wp.transform(
            wp.vec3(0.0, 0.0, 0.1),
            wp.quat_identity(),
        )

        world_builder.add_urdf(
            allegro_urdf,
            floating=False,
            xform=allegro_xform,
            enable_self_collisions=False,
        )

        self.allegro_body_offset = body_offset
        self.allegro_body_count = world_builder.body_count - body_offset
        self.allegro_dof_count = world_builder.joint_dof_count - dof_offset
        self.allegro_dof_start = dof_offset

        # Connect Allegro to Kuka EE
        allegro_root_joint_idx = joint_offset
        if world_builder.joint_parent[allegro_root_joint_idx] == -1:
            world_builder.joint_parent[allegro_root_joint_idx] = self.ee_body_idx
            world_builder.joint_X_p[allegro_root_joint_idx] = allegro_xform

        # Merge articulations
        for j in range(joint_offset, world_builder.joint_count):
            world_builder.joint_articulation[j] = 0
        world_builder.articulation_start = [0]
        world_builder.articulation_label = ["kuka_allegro"]
        world_builder.articulation_world = [0]

        # Allegro joint parameters
        for i in range(self.allegro_dof_start, world_builder.joint_dof_count):
            world_builder.joint_target_ke[i] = 200.0
            world_builder.joint_target_kd[i] = 20.0
            world_builder.joint_effort_limit[i] = 20.0
            world_builder.joint_armature[i] = 0.05
            world_builder.joint_target_mode[i] = int(JointTargetMode.POSITION)

        # Initial Allegro pose (open hand)
        allegro_init_q = [0.0, 0.2, 0.2, 0.2] * 4
        world_builder.joint_q[self.allegro_dof_start:self.allegro_dof_start + self.allegro_dof_count] = allegro_init_q
        world_builder.joint_target_pos[self.allegro_dof_start:self.allegro_dof_start + self.allegro_dof_count] = allegro_init_q

        # Store joint limits (from single env, same for all)
        self.kuka_joint_lower = np.array(world_builder.joint_limit_lower[:self.kuka_dof_count], dtype=np.float32)
        self.kuka_joint_upper = np.array(world_builder.joint_limit_upper[:self.kuka_dof_count], dtype=np.float32)
        self.allegro_joint_lower = np.array(
            world_builder.joint_limit_lower[self.allegro_dof_start:self.allegro_dof_start + self.allegro_dof_count],
            dtype=np.float32,
        )
        self.allegro_joint_upper = np.array(
            world_builder.joint_limit_upper[self.allegro_dof_start:self.allegro_dof_start + self.allegro_dof_count],
            dtype=np.float32,
        )

        # === Add cube (free-floating body) ===
        cube_size = 0.05
        cube_cfg = newton.ModelBuilder.ShapeConfig(
            ke=1.0e4, kd=1.0e3, mu=1.0,
            density=0.1 / (cube_size ** 3),
        )
        self.table_surface_z = 0.21
        self.cube_spawn_pos = np.array([0.0, -0.62, self.table_surface_z + cube_size / 2 + 0.01])
        cube_xform = wp.transform(wp.vec3(*self.cube_spawn_pos), wp.quat_identity())
        self.cube_body_idx_local = world_builder.add_body(xform=cube_xform, label="cube")
        world_builder.add_shape_box(
            body=self.cube_body_idx_local,
            xform=wp.transform_identity(),
            hx=cube_size / 2,
            hy=cube_size / 2,
            hz=cube_size / 2,
            cfg=cube_cfg,
        )

        self.bodies_per_world = world_builder.body_count
        self.dofs_per_world = world_builder.joint_dof_count

        # === Replicate into scene ===
        self.env_spacing = (2.0, 2.0, 0.0)
        scene = newton.ModelBuilder()
        newton.solvers.SolverMuJoCo.register_custom_attributes(scene)
        scene.replicate(world_builder, world_count=self.world_count, spacing=self.env_spacing)

        # Ground plane (shared)
        scene.add_ground_plane()

        # === Add tables per world (static shapes on body=-1 are not replicated) ===
        table_cfg = newton.ModelBuilder.ShapeConfig(ke=1.0e5, kd=1.0e3, mu=0.8)
        world_offsets = compute_world_offsets(self.world_count, self.env_spacing, up_axis=scene.up_axis)
        for env_idx in range(self.world_count):
            ox, oy, oz = world_offsets[env_idx]
            table_xform = wp.transform(
                wp.vec3(ox + 0.0, oy - 0.62, oz + 0.20),
                wp.quat_identity(),
            )
            scene.add_shape_box(
                body=-1,
                xform=table_xform,
                hx=0.3, hy=0.3, hz=0.01,
                cfg=table_cfg,
            )

        self.model = scene.finalize()
        print(f"[INFO] Scene: {self.model.body_count} bodies, {self.model.joint_count} joints, "
              f"{self.model.joint_dof_count} DOFs, {self.world_count} worlds")
        print(f"[INFO] Per world: {self.bodies_per_world} bodies, {self.dofs_per_world} DOFs "
              f"(Kuka: {self.kuka_dof_count}, Allegro: {self.allegro_dof_count})")

    def _setup_control(self):
        """Setup scripted policy phases."""
        self.anim_time = 0.0
        self.frame_count = 0
        self.phase = "init"

        # Kuka poses (joint0 = -π/2 to face -Y)
        self.kuka_init = np.array([-math.pi / 2, 0.4, 0.0, -1.4, 0.0, 1.2, 0.0], dtype=np.float32)
        self.kuka_reach = np.array([-math.pi / 2, 0.9, 0.0, -1.2, 0.0, 1.9, 0.0], dtype=np.float32)
        self.kuka_lift = np.array([-math.pi / 2, 0.3, 0.0, -1.5, 0.0, 1.2, 0.0], dtype=np.float32)

        # Allegro poses
        self.allegro_open = np.array([0.0, 0.1, 0.1, 0.1] * 4, dtype=np.float32)
        self.allegro_closed = np.array([0.0, 1.0, 1.0, 1.0] * 3 + [1.5, 0.6, 0.3, 0.6], dtype=np.float32)
        self.allegro_closed = np.clip(self.allegro_closed, self.allegro_joint_lower, self.allegro_joint_upper)

    def simulate(self):
        """Run simulation substeps."""
        self.state_0.clear_forces()
        self.state_1.clear_forces()

        for i in range(self.sim_substeps):
            if i % 2 == 0:
                self.contacts = self.model.collide(self.state_0)
            self.solver.step(self.state_0, self.state_1, self.control, self.contacts, self.sim_dt)
            self.state_0, self.state_1 = self.state_1, self.state_0

    def step(self):
        """Step with scripted reach -> grasp -> lift policy."""
        self.anim_time += self.frame_dt
        self.frame_count += 1

        # Phase transitions: reach(0-3s) -> grasp(3-5s) -> lift(5-8s) -> loop
        if self.anim_time < 3.0:
            self.phase = "reach"
            t = min(self.anim_time / 3.0, 1.0)
            kuka_target = (1 - t) * self.kuka_init + t * self.kuka_reach
            allegro_target = self.allegro_open
        elif self.anim_time < 5.0:
            self.phase = "grasp"
            t = min((self.anim_time - 3.0) / 2.0, 1.0)
            kuka_target = self.kuka_reach
            allegro_target = (1 - t) * self.allegro_open + t * self.allegro_closed
        elif self.anim_time < 8.0:
            self.phase = "lift"
            t = min((self.anim_time - 5.0) / 3.0, 1.0)
            kuka_target = (1 - t) * self.kuka_reach + t * self.kuka_lift
            allegro_target = self.allegro_closed
        else:
            self.anim_time = 0.0
            self.phase = "reach"
            kuka_target = self.kuka_init
            allegro_target = self.allegro_open

        # Apply same control to all worlds
        control_np = self.control.joint_target_pos.numpy()
        for w in range(self.world_count):
            offset = w * self.dofs_per_world
            control_np[offset:offset + self.kuka_dof_count] = kuka_target
            control_np[offset + self.allegro_dof_start:offset + self.allegro_dof_start + self.allegro_dof_count] = allegro_target
        self.control.joint_target_pos = wp.array(control_np, dtype=wp.float32, device=self.device)

        self.simulate()

        # Log status every second (world 0 only)
        if self.frame_count % 60 == 0:
            body_q = self.state_0.body_q.numpy()  # (num_bodies, 7)
            cube_pos = body_q[self.cube_body_idx_local, :3]
            ee_pos = body_q[self.ee_body_idx, :3]
            dist = np.linalg.norm(ee_pos - cube_pos)
            print(f"[t={self.anim_time:.1f}s] phase={self.phase}, "
                  f"cube_z={cube_pos[2]:.3f}, ee_dist={dist:.3f}, "
                  f"ee=({ee_pos[0]:.2f}, {ee_pos[1]:.2f}, {ee_pos[2]:.2f})")

        self.sim_time += self.frame_dt

    def render(self):
        """Render the scene."""
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        self.viewer.log_contacts(self.contacts, self.state_0)
        self.viewer.end_frame()

    def test_final(self):
        """Validation test - check simulation ran without NaN."""
        body_q = self.state_0.body_q.numpy()
        assert not np.any(np.isnan(body_q)), "NaN detected in body positions"


if __name__ == "__main__":
    parser = newton.examples.create_parser()
    newton.examples.add_world_count_arg(parser)

    viewer, args = newton.examples.init(parser)
    example = Example(viewer, args.world_count)
    newton.examples.run(example, args)
