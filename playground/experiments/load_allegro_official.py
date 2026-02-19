# Allegro Hand 로드 (공식 원격 에셋)
#
# 실행: uv run python playground/experiments/load_allegro_official.py
#
# 공식 원격 에셋(wonik_allegro)을 다운로드해서 사용합니다.

import warp as wp

import newton
import newton.examples
from newton import ActuatorMode


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

        # 기본 설정
        builder.default_shape_cfg.ke = 1.0e3
        builder.default_shape_cfg.kd = 1.0e2

        # 공식 Allegro 에셋 다운로드
        print("\n[Allegro 에셋 다운로드 중...]")
        asset_path = newton.utils.download_asset("wonik_allegro")
        asset_file = str(asset_path / "usd" / "allegro_left_hand_with_cube.usda")
        print(f"  경로: {asset_file}")

        # Allegro Hand 로드 (z=0.3m에 고정)
        builder.add_usd(
            asset_file,
            xform=wp.transform(wp.vec3(0, 0, 0.3), wp.quat_identity()),
            collapse_fixed_joints=True,
            enable_self_collisions=False,
        )

        # 조인트에 위치 제어(PD) 활성화
        for i in range(builder.joint_dof_count):
            builder.joint_target_ke[i] = 150.0  # P 게인
            builder.joint_target_kd[i] = 5.0    # D 게인
            builder.joint_target_pos[i] = 0.0   # 목표 위치
            builder.joint_act_mode[i] = int(ActuatorMode.POSITION)

        builder.add_ground_plane()
        self.model = builder.finalize()

        # MuJoCo 솔버 사용
        self.solver = newton.solvers.SolverMuJoCo(self.model)

        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()

        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.state_0)

        self.contacts = self.model.collide(self.state_0)

        self.viewer.set_model(self.model)

        print(f"\n[Allegro Hand (공식 에셋)]")
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
