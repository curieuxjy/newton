# 나의 첫 번째 Newton 시뮬레이션
#
# 실행: uv run python playground/experiments/my_first_sim.py
#
# 원본 참고: newton/examples/basic/example_basic_pendulum.py

import warp as wp

import newton
import newton.examples


class Example:
    def __init__(self, viewer, args=None):
        # 시뮬레이션 파라미터 설정
        self.fps = 60
        self.frame_dt = 1.0 / self.fps
        self.sim_time = 0.0
        self.sim_substeps = 10
        self.sim_dt = self.frame_dt / self.sim_substeps

        self.viewer = viewer
        self.args = args

        # 모델 빌더 생성
        builder = newton.ModelBuilder()

        # 접촉 파라미터 설정 (Newton 2026-02-09 이후 기본값: ke=2.5e3)
        builder.default_shape_cfg.ke = 2.5e3  # 접촉 강성
        builder.default_shape_cfg.kd = 250.0  # 접촉 감쇠

        # TODO: 여기에 시뮬레이션 객체를 추가하세요
        # 예시: 간단한 박스 추가
        link_0 = builder.add_link()
        builder.add_shape_box(link_0, hx=0.5, hy=0.5, hz=0.5)

        # 조인트 추가 (공중에 매달기)
        j0 = builder.add_joint_revolute(
            parent=-1,  # -1은 월드(고정점)
            child=link_0,
            axis=wp.vec3(0.0, 1.0, 0.0),
            parent_xform=wp.transform(p=wp.vec3(0.0, 0.0, 3.0), q=wp.quat_identity()),
            child_xform=wp.transform(p=wp.vec3(-1.0, 0.0, 0.0), q=wp.quat_identity()),
        )

        builder.add_articulation([j0], key="my_pendulum")

        # 바닥 추가
        builder.add_ground_plane()

        # 모델 확정
        self.model = builder.finalize()

        # 솔버 생성
        self.solver = newton.solvers.SolverXPBD(self.model)

        # 상태 초기화
        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()

        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.state_0)

        # 충돌 파이프라인 생성
        self.collision_pipeline = newton.examples.create_collision_pipeline(self.model, self.args)
        self.contacts = self.model.collide(self.state_0, collision_pipeline=self.collision_pipeline)

        self.viewer.set_model(self.model)

    def simulate(self):
        for _ in range(self.sim_substeps):
            self.state_0.clear_forces()
            self.viewer.apply_forces(self.state_0)
            self.contacts = self.model.collide(self.state_0, collision_pipeline=self.collision_pipeline)
            self.solver.step(self.state_0, self.state_1, self.control, self.contacts, self.sim_dt)
            self.state_0, self.state_1 = self.state_1, self.state_0

    def step(self):
        self.simulate()
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
