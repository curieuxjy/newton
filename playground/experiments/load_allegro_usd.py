# Allegro Hand 로드 (USD)
#
# 실행: uv run python playground/experiments/load_allegro_usd.py

import warp as wp

import newton
import newton.examples

ASSET_PATH = "playground/asset/allegro_hand_right.usd"


class Example:
    def __init__(self, viewer, args=None):
        self.fps = 60
        self.frame_dt = 1.0 / self.fps
        self.sim_time = 0.0
        self.sim_substeps = 10
        self.sim_dt = self.frame_dt / self.sim_substeps
        self.viewer = viewer
        self.args = args

        builder = newton.ModelBuilder()

        # MuJoCo 솔버용 커스텀 속성 등록
        newton.solvers.SolverMuJoCo.register_custom_attributes(builder)

        # Allegro Hand USD 로드 (z=0.3m에 고정)
        builder.add_usd(
            ASSET_PATH,
            xform=wp.transform(wp.vec3(0, 0, 0.3), wp.quat_identity()),
            collapse_fixed_joints=True,
            enable_self_collisions=False,
        )

        builder.add_ground_plane()
        self.model = builder.finalize()

        self.solver = newton.solvers.SolverXPBD(self.model)

        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()

        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.state_0)

        self.contacts = self.model.collide(self.state_0)

        self.viewer.set_model(self.model)

        print(f"\n[Allegro Hand USD]")
        print(f"  파일: {ASSET_PATH}")
        print(f"  링크 수: {self.model.body_count}")
        print(f"  조인트 수: {self.model.joint_count}")
        print(f"  DOF: {len(self.model.joint_q)}\n")

    def step(self):
        for _ in range(self.sim_substeps):
            self.state_0.clear_forces()
            self.viewer.apply_forces(self.state_0)
            self.contacts = self.model.collide(self.state_0)
            self.solver.step(self.state_0, self.state_1, self.control, self.contacts, self.sim_dt)
            self.state_0, self.state_1 = self.state_1, self.state_0
        self.sim_time += self.frame_dt

    def render(self):
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        self.viewer.log_contacts(self.contacts, self.state_0)
        self.viewer.end_frame()


if __name__ == "__main__":
    viewer, args = newton.examples.init()
    example = Example(viewer, args)
    newton.examples.run(example, args)
