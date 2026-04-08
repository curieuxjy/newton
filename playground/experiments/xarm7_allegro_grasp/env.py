"""xArm7 + Allegro Hand Cube Grasping Environment.

Subclasses :class:`FrankaAllegroGraspEnv` and overrides only ``_build_simulation``
to swap the Franka FR3 URDF for the UFactory xArm7 MJCF (mujoco_menagerie).

Internally the parent class still uses field names ``franka_dof_count``,
``franka_joint_lower``, etc. — those refer to the arm in general here.
"""

from __future__ import annotations

import numpy as np
import torch
import warp as wp

import newton
from newton import JointTargetMode

from playground.experiments.franka_allegro_grasp.env import FrankaAllegroGraspEnv

from ._menagerie import download_menagerie_asset
from ._xarm7_fabric import XArm7FabricActionController


class XArm7AllegroGraspEnv(FrankaAllegroGraspEnv):
    """Vectorized environment for xArm7 + Allegro cube grasping task.

    Identical task definition to :class:`FrankaAllegroGraspEnv` — only the
    arm asset and a few arm-specific constants change.
    """

    # Default arm joint configuration: arm extended forward over the table.
    # xArm7 joints (standard UFactory ordering): joint1..joint7.
    # These are an initial guess; tune after first visualize run.
    XARM7_INIT_Q = [0.0, -0.5, 0.0, 0.5, 0.0, 1.0, 0.0]

    # Substring used to find the end-effector body in the parsed MJCF.
    # ufactory_xarm7 ends in ``link7``; the flange/EE is attached to it.
    XARM7_EE_LINK_KEY = "link7"

    # Mount transform of the Allegro hand relative to the xArm7 EE link.
    # Same offset as Franka for now (10 cm along link Z, 180° about Z).
    ALLEGRO_MOUNT_XFORM = (
        wp.vec3(0.0, 0.0, 0.1),
        # quat populated lazily because wp.quat_from_axis_angle isn't
        # constant-foldable at class-body eval time.
    )

    def __init__(self, config, device: str = "cuda", headless: bool = True):
        super().__init__(config, device=device, headless=headless)
        # Swap the parent's Franka FABRICS controller for the xArm7 variant.
        # ``GraspFabric`` (self.fabric) takes ee_pos/ee_quat from sim state and
        # is arm-agnostic — only the IK controller needs xArm7 kinematics.
        if self.use_fabric_actions and self.fabric_action_controller is not None:
            self.fabric_action_controller = XArm7FabricActionController(
                franka_dof=self.franka_dof_count,
                allegro_dof=self.allegro_dof_count,
                franka_joint_lower=self.franka_joint_lower,
                franka_joint_upper=self.franka_joint_upper,
                allegro_joint_lower=self.allegro_joint_lower,
                allegro_joint_upper=self.allegro_joint_upper,
                device=self.torch_device,
                damping=config.fabric_ik_damping,
                ik_step_size=config.fabric_ik_step_size,
            )

    def reset(self, env_ids=None):
        result = super().reset(env_ids)
        # Parent uses ``table_height + lift_height`` for goal z, but in this
        # codebase the table top sits at ``table_height/2 + table_size[2]/2``,
        # so the parent's goal floats too high. Recompute the lift target
        # relative to the cube's actual resting position.
        body_q = torch.from_numpy(self.state_0.body_q.numpy()).to(self.torch_device)
        cube_pos = body_q.reshape(self.num_envs, self.bodies_per_env, 7)[:, self.cube_body_idx, :3]
        lift_z = cube_pos[:, 2] + self.config.lift_height
        if env_ids is None:
            self.goal_pos[:, 0] = cube_pos[:, 0]
            self.goal_pos[:, 1] = cube_pos[:, 1]
            self.goal_pos[:, 2] = lift_z
        else:
            self.goal_pos[env_ids, 0] = cube_pos[env_ids, 0]
            self.goal_pos[env_ids, 1] = cube_pos[env_ids, 1]
            self.goal_pos[env_ids, 2] = lift_z[env_ids]
        return result

    def _build_simulation(self):
        """Build the Newton simulation with xArm7 + Allegro, table, and cube."""
        single_env_builder = newton.ModelBuilder()
        newton.solvers.SolverMuJoCo.register_custom_attributes(single_env_builder)

        # Contact parameters (same as Franka variant)
        single_env_builder.default_shape_cfg.ke = 1.0e4
        single_env_builder.default_shape_cfg.kd = 1.0e3
        single_env_builder.default_shape_cfg.mu = 1.2

        # === Load xArm7 arm (mujoco_menagerie) ===
        xarm_asset = download_menagerie_asset("ufactory_xarm7")
        # Prefer the bare robot MJCF over scene.xml so we don't pick up
        # menagerie-provided ground/lights/cameras.
        # Prefer the no-hand variant — the default xarm7.xml/scene.xml include
        # the xArm gripper (~6 extra DOFs) which would conflict with the
        # Allegro hand and break the parent's 7+16=23 joint-target layout.
        candidate_files = ["xarm7_nohand.xml", "xarm7.xml", "scene.xml"]
        mjcf_path = None
        for name in candidate_files:
            if (xarm_asset / name).exists():
                mjcf_path = xarm_asset / name
                break
        if mjcf_path is None:
            raise FileNotFoundError(
                f"Could not find xarm7.xml or scene.xml in {xarm_asset}"
            )

        # Translate robot so EE is centered on table's Y axis (same as Franka).
        robot_pos = wp.vec3(0.0, 0.25, 0.0)
        single_env_builder.add_mjcf(
            str(mjcf_path),
            xform=wp.transform(robot_pos, wp.quat_identity()),
            enable_self_collisions=False,
            ignore_inertial_definitions=False,
            ctrl_direct=True,
        )

        # ``franka_*`` field names below refer to the arm in general — here xArm7.
        self.franka_body_count = single_env_builder.body_count
        self.franka_joint_count = single_env_builder.joint_count
        self.franka_dof_count = single_env_builder.joint_dof_count

        if self.franka_dof_count != 7:
            print(
                f"[xarm7_allegro_grasp] Warning: expected 7 arm DOFs, "
                f"got {self.franka_dof_count}. Check the loaded MJCF."
            )

        # Set arm joint parameters (config field names retained from parent)
        for i in range(self.franka_dof_count):
            single_env_builder.joint_target_ke[i] = self.config.franka_stiffness
            single_env_builder.joint_target_kd[i] = self.config.franka_damping
            single_env_builder.joint_effort_limit[i] = self.config.franka_effort_limit
            single_env_builder.joint_armature[i] = self.config.franka_armature
            single_env_builder.joint_target_mode[i] = int(JointTargetMode.POSITION)

        # Initial xArm7 pose
        init_q = list(self.XARM7_INIT_Q)
        if len(init_q) != self.franka_dof_count:
            init_q = [0.0] * self.franka_dof_count
        single_env_builder.joint_q[: self.franka_dof_count] = init_q
        single_env_builder.joint_target_pos[: self.franka_dof_count] = init_q

        # Find end-effector body
        self.ee_body_idx = -1
        for i, key in enumerate(single_env_builder.body_label):
            if self.XARM7_EE_LINK_KEY in key:
                self.ee_body_idx = i
        if self.ee_body_idx < 0:
            self.ee_body_idx = self.franka_body_count - 1

        # === Load Allegro hand (URDF, same as Franka variant) ===
        allegro_asset = newton.utils.download_asset("wonik_allegro")
        allegro_urdf = allegro_asset / "urdf/allegro_hand_description_left.urdf"

        allegro_body_offset = single_env_builder.body_count
        allegro_joint_offset = single_env_builder.joint_count
        allegro_dof_offset = single_env_builder.joint_dof_count

        allegro_xform = wp.transform(
            self.ALLEGRO_MOUNT_XFORM[0],
            wp.quat_from_axis_angle(wp.vec3(0.0, 0.0, 1.0), float(np.pi)),
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

        # Connect Allegro to xArm7 EE
        allegro_root_joint_idx = allegro_joint_offset
        if single_env_builder.joint_parent[allegro_root_joint_idx] == -1:
            single_env_builder.joint_parent[allegro_root_joint_idx] = self.ee_body_idx
            single_env_builder.joint_X_p[allegro_root_joint_idx] = allegro_xform

        # Merge into one articulation
        for j in range(allegro_joint_offset, single_env_builder.joint_count):
            single_env_builder.joint_articulation[j] = 0
        single_env_builder.articulation_start = [0]
        single_env_builder.articulation_label = ["xarm7_allegro"]
        single_env_builder.articulation_world = [0]

        # Allegro joint parameters
        for i in range(self.allegro_dof_start, single_env_builder.joint_dof_count):
            single_env_builder.joint_target_ke[i] = self.config.hand_stiffness
            single_env_builder.joint_target_kd[i] = self.config.hand_damping
            single_env_builder.joint_effort_limit[i] = self.config.hand_effort_limit
            single_env_builder.joint_armature[i] = self.config.hand_armature
            single_env_builder.joint_target_mode[i] = int(JointTargetMode.POSITION)

        allegro_init_q = [0.0, 0.3, 0.3, 0.3] * 4
        single_env_builder.joint_q[
            self.allegro_dof_start : self.allegro_dof_start + self.allegro_dof_count
        ] = allegro_init_q
        single_env_builder.joint_target_pos[
            self.allegro_dof_start : self.allegro_dof_start + self.allegro_dof_count
        ] = allegro_init_q

        # Joint limits
        self.franka_joint_lower = torch.tensor(
            single_env_builder.joint_limit_lower[: self.franka_dof_count],
            dtype=torch.float32,
            device=self.torch_device,
        )
        self.franka_joint_upper = torch.tensor(
            single_env_builder.joint_limit_upper[: self.franka_dof_count],
            dtype=torch.float32,
            device=self.torch_device,
        )
        self.allegro_joint_lower = torch.tensor(
            single_env_builder.joint_limit_lower[
                self.allegro_dof_start : self.allegro_dof_start + self.allegro_dof_count
            ],
            dtype=torch.float32,
            device=self.torch_device,
        )
        self.allegro_joint_upper = torch.tensor(
            single_env_builder.joint_limit_upper[
                self.allegro_dof_start : self.allegro_dof_start + self.allegro_dof_count
            ],
            dtype=torch.float32,
            device=self.torch_device,
        )

        self.allegro_joint_lower = torch.clamp(self.allegro_joint_lower, min=-3.14)
        self.allegro_joint_upper = torch.clamp(self.allegro_joint_upper, max=3.14)

        self.joint_lower = torch.cat([self.franka_joint_lower, self.allegro_joint_lower])
        self.joint_upper = torch.cat([self.franka_joint_upper, self.allegro_joint_upper])
        self.joint_range = self.joint_upper - self.joint_lower
        self.joint_mid = (self.joint_upper + self.joint_lower) / 2

        self.total_robot_bodies = single_env_builder.body_count

        # === The remainder of this method mirrors FrankaAllegroGraspEnv. ===
        # Cube as free-floating body (positions corrected in reset/_store_initial_state)
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
        cube_body_idx = single_env_builder.add_body(xform=cube_xform, label="cube")
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

        builder.default_shape_cfg.ke = 1.0e4
        builder.default_shape_cfg.kd = 1.0e3
        builder.add_ground_plane()

        # Tables (static — added after replication)
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

        self.joint_q_per_env = self.model.joint_q.shape[0] // self.num_envs
        self.joint_qd_per_env = self.model.joint_qd.shape[0] // self.num_envs
        self.bodies_per_env = self.model.body_q.shape[0] // self.num_envs

        print(f"[xarm7_allegro_grasp] joint_q total: {self.model.joint_q.shape[0]}, per_env: {self.joint_q_per_env}")
        print(f"[xarm7_allegro_grasp] arm_dof: {self.franka_dof_count}, allegro_dof: {self.allegro_dof_count}")

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

        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.state_0)
