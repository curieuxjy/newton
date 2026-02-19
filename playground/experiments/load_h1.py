# Unitree H1 휴머노이드 로드 (원격 다운로드)
#
# 실행: uv run python playground/experiments/load_h1.py
#
# 참고: 처음 실행 시 에셋을 다운로드합니다 (이후 캐시 사용)

import warp as wp

import newton
import newton.examples


class Example:
    def __init__(self, viewer, args=None):
        self.fps = 50
        self.frame_dt = 1.0 / self.fps
        self.sim_time = 0.0
        self.sim_substeps = 10
        self.sim_dt = self.frame_dt / self.sim_substeps
        self.viewer = viewer
        self.args = args

        builder = newton.ModelBuilder()

        # MuJoCo 솔버용 커스텀 속성 등록
        newton.solvers.SolverMuJoCo.register_custom_attributes(builder)

        # 접촉 파라미터 설정 (Newton 2026-02-09 기본값, MuJoCo 정렬)
        builder.default_shape_cfg.ke = 2.5e3  # 접촉 강성
        builder.default_shape_cfg.kd = 250.0  # 접촉 감쇠

        # H1 에셋 다운로드 (자동 캐시)
        print("\n[H1 에셋 다운로드 중...]")
        asset_path = newton.utils.download_asset("unitree_h1")
        usd_file = str(asset_path / "usd" / "h1_minimal.usda")
        print(f"  경로: {usd_file}")

        # H1 USD 로드
        builder.add_usd(
            usd_file,
            ignore_paths=["/GroundPlane"],
            collapse_fixed_joints=False,
            enable_self_collisions=False,
            hide_collision_shapes=True,
        )

        # 충돌 메시 단순화 (성능 향상)
        builder.approximate_meshes("bounding_box")

        builder.add_ground_plane()
        self.model = builder.finalize()

        # MuJoCo 솔버 사용 (H1에 최적화)
        self.solver = newton.solvers.SolverMuJoCo(self.model)

        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()

        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.state_0)

        self.contacts = self.model.collide(self.state_0)

        self.viewer.set_model(self.model)

        print(f"\n[Unitree H1 Humanoid]")
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
