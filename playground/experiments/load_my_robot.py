# 내 로봇 모델 불러오기 템플릿
#
# 실행: uv run python playground/experiments/load_my_robot.py
#
# 사용법:
#   1. 아래 "로봇 로드" 섹션에서 원하는 옵션 주석 해제
#   2. 경로를 내 로봇 파일로 변경
#   3. 실행

import warp as wp

import newton
import newton.examples


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

        # ============================================================
        # 로봇 로드 - 아래 옵션 중 하나를 선택하세요
        # ============================================================

        # ----- 옵션 1: 내장 URDF (quadruped) -----
        builder.add_urdf(
            newton.examples.get_asset("quadruped.urdf"),
            xform=wp.transform(wp.vec3(0, 0, 0.5), wp.quat_identity()),
            floating=True,
            enable_self_collisions=False,
        )

        # ----- 옵션 2: 내 URDF 파일 -----
        # builder.add_urdf(
        #     "/path/to/my_robot.urdf",
        #     xform=wp.transform(wp.vec3(0, 0, 1), wp.quat_identity()),
        #     floating=True,       # True: 자유 이동, False: 바닥 고정
        #     scale=1.0,           # 스케일 (미터 단위)
        #     enable_self_collisions=False,
        # )

        # ----- 옵션 3: 내장 MJCF (humanoid) -----
        # builder.add_mjcf(
        #     newton.examples.get_asset("nv_humanoid.xml"),
        #     xform=wp.transform(wp.vec3(0, 0, 1.3), wp.quat_identity()),
        #     ignore_names=["floor", "ground"],
        # )

        # ----- 옵션 4: 원격 에셋 다운로드 (Unitree H1) -----
        # newton.solvers.SolverMuJoCo.register_custom_attributes(builder)
        # asset_path = newton.utils.download_asset("unitree_h1")
        # builder.add_usd(
        #     str(asset_path / "usd" / "h1_minimal.usda"),
        #     ignore_paths=["/GroundPlane"],
        #     collapse_fixed_joints=False,
        #     enable_self_collisions=False,
        # )
        # builder.approximate_meshes("bounding_box")

        # ----- 옵션 5: 원격 에셋 다운로드 (Franka Panda) -----
        # asset_path = newton.utils.download_asset("franka_emika_panda")
        # builder.add_urdf(
        #     str(asset_path / "panda.urdf"),
        #     xform=wp.transform(wp.vec3(0, 0, 0), wp.quat_identity()),
        #     floating=False,  # 로봇 팔은 바닥 고정
        # )

        # ============================================================

        # 바닥 추가
        builder.add_ground_plane()

        # 모델 확정
        self.model = builder.finalize()

        # 솔버 선택
        # - SolverXPBD: 범용, 안정적
        # - SolverMuJoCo: MuJoCo 호환, 고급 기능
        self.solver = newton.solvers.SolverXPBD(self.model)

        # 상태 초기화
        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()

        # 순운동학 계산 (초기 포즈 설정)
        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.state_0)

        # 충돌 감지
        self.contacts = self.model.collide(self.state_0)

        self.viewer.set_model(self.model)

        # 모델 정보 출력
        self.print_model_info()

    def print_model_info(self):
        """로드된 모델 정보 출력"""
        print("\n" + "=" * 50)
        print("로드된 모델 정보")
        print("=" * 50)
        print(f"링크(바디) 수: {self.model.body_count}")
        print(f"조인트 수: {self.model.joint_count}")
        print(f"셰이프 수: {self.model.shape_count}")
        print(f"자유도(DOF): {len(self.model.joint_q)}")
        print("=" * 50 + "\n")

    def simulate(self):
        for _ in range(self.sim_substeps):
            self.state_0.clear_forces()
            self.viewer.apply_forces(self.state_0)
            self.contacts = self.model.collide(self.state_0)
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
