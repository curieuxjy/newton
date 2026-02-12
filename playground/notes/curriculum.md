# Newton 학습 커리큘럼

Humanoid, Manipulator, Sensor 예제를 단계별로 학습하는 커리큘럼입니다.

---

## 학습 트랙 개요

| 단계 | Manipulator | Humanoid | Sensor |
|------|-------------|----------|--------|
| 1 | ik_franka (IK 기초) | - | sensor_imu (기본 센서) |
| 2 | robot_ur10 (다중 월드) | robot_humanoid (MJCF) | sensor_contact (접촉 감지) |
| 3 | robot_allegro_hand (복잡 관절) | robot_h1 (USD, H1) | sensor_tiled_camera (카메라) |
| 4 | robot_panda_hydro (IK+물리+조작) | robot_g1 (고급 솔버) | - |

---

## Week 1: 기초 다지기

### Day 1-2: IK 기초 + IMU 센서

**1. example_ik_franka** (Beginner)
```bash
uv run -m newton.examples ik_franka
```
- Franka FR3 로봇 팔의 역운동학(IK)
- 기즈모로 타겟 위치 직접 조작
- 물리 시뮬레이션 없이 순수 IK만

**핵심 개념:**
- `ik.IKSolver()` - IK 솔버 생성
- `ik.IKPositionObjective` - 위치 목표
- `ik.IKRotationObjective` - 회전 목표
- `newton.eval_fk()` - 순운동학 계산

**실습:** 타겟 위치를 바꿔보며 로봇 팔이 어떻게 움직이는지 관찰

---

**2. example_sensor_imu** (Intermediate)
```bash
uv run -m newton.examples sensor_imu
```
- IMU(관성측정장치) 센서로 가속도 측정
- 세 개의 큐브가 떨어지며 가속도에 따라 색상 변화

**핵심 개념:**
- `SensorIMU` - IMU 센서 클래스
- `imu.update()` - 센서 데이터 갱신
- Warp 커널로 GPU 가속 데이터 처리

**실습:** 가속도 값이 중력(9.8)과 어떻게 관련되는지 확인

---

### Day 3-4: 다중 월드 시뮬레이션

**3. example_robot_ur10** (Intermediate)
```bash
uv run -m newton.examples robot_ur10
```
- UR10 협동로봇 팔 시뮬레이션
- 16~100개 월드 동시 시뮬레이션
- 사인파 궤적으로 관절 제어

**핵심 개념:**
- `ModelBuilder.add_usd()` - USD 파일 로드
- `ArticulationView` - 다중 월드에서 로봇 일괄 접근
- `ActuatorMode.POSITION` - 위치 제어
- `.replicate()` - 월드 복제

**실습:** `num_envs` 값을 바꿔보며 성능 차이 관찰

---

**4. example_robot_humanoid** (Intermediate)
```bash
uv run -m newton.examples robot_humanoid
```
- MJCF 포맷의 휴머노이드 로봇
- 다중 월드 시뮬레이션
- GPU 그래프 캡처로 성능 최적화

**핵심 개념:**
- `ModelBuilder.add_mjcf()` - MJCF 파일 로드
- `SolverMuJoCo` - MuJoCo 솔버
- CUDA 그래프 캡처

**실습:** humanoid가 바닥에 쓰러지는 동작 관찰

---

## Week 2: 중급 과정

### Day 5-6: 접촉 센서 + 복잡한 로봇

**5. example_sensor_contact** (Intermediate)
```bash
uv run -m newton.examples sensor_contact
```
- 접촉력 센서로 물체 간 힘 측정
- 정규식으로 특정 물체 쌍 필터링
- ImGui로 실시간 그래프 표시

**핵심 개념:**
- `SensorContact` - 접촉력 센서
- `populate_contacts()` - 접촉 데이터 수집
- `net_force` - 총 접촉력 조회
- 정규식 패턴 매칭

**실습:** 어떤 물체들이 접촉하는지 로그로 확인

---

**6. example_robot_allegro_hand** (Intermediate)
```bash
uv run -m newton.examples robot_allegro_hand
```
- Allegro 다지 손(dexterous hand) 시뮬레이션
- 루트 트랜스폼 애니메이션
- 런타임 모델 속성 변경

**핵심 개념:**
- 복잡한 관절 구조 (16+ DOF)
- `SolverNotifyFlags.JOINT_PROPERTIES` - 솔버에 변경 알림
- `joint_parent_xform` - 루트 위치 제어

**실습:** 손가락 관절 목표값을 바꿔보기

---

### Day 7-8: 실제 휴머노이드 로봇

**7. example_robot_h1** (Intermediate-Advanced)
```bash
uv run -m newton.examples robot_h1
```
- Unitree H1 휴머노이드 (USD 포맷)
- 메시 근사로 충돌 최적화
- MuJoCo 커스텀 속성 등록

**핵심 개념:**
- `approximate_meshes("bounding_box")` - 충돌 메시 단순화
- `SolverMuJoCo.register_custom_attributes()` - 커스텀 속성
- 월드 오프셋 시각화

**실습:** H1이 서 있다가 쓰러지는 과정 관찰

---

**8. example_robot_g1** (Advanced)
```bash
uv run -m newton.examples robot_g1
```
- Unitree G1 휴머노이드 (더 복잡한 모델)
- Newton 솔버 + Implicit Fast 적분기
- 선택적 관절 제어 (조인트 6번 이후만)

**핵심 개념:**
- Newton 솔버 (고급 수치 방법)
- `implicit_fast` 적분기
- 타원 마찰 원뿔 (elliptic friction cone)
- 선택적 액추에이터 설정

**실습:** G1과 H1의 동작 차이 비교

---

## Week 3: 고급 과정

### Day 9-10: 카메라 센서 + 조작 작업

**9. example_sensor_tiled_camera** (Advanced)
```bash
uv run -m newton.examples sensor_tiled_camera
```
- 24개 월드 동시 렌더링 (6x4 그리드)
- RGB, Depth, Normal, Semantic 출력
- GPU-OpenGL 연동

**핵심 개념:**
- `SensorTiledCamera` - 타일 카메라 센서
- `create_color_image_output()`, `create_depth_image_output()`
- `compute_pinhole_camera_rays()` - 카메라 광선 계산
- Semantic segmentation

**실습:** 각 렌더 모드(RGB/Depth/Normal) 전환해보기

---

**10. example_robot_panda_hydro** (Advanced)
```bash
uv run -m newton.examples robot_panda_hydro
```
- Franka Panda로 pick-and-place 작업
- Hydroelastic 접촉 (소프트바디 상호작용)
- IK + 물리 시뮬레이션 통합

**핵심 개념:**
- `SDFHydroelasticConfig` - Hydroelastic 접촉
- `CollisionPipeline` + `BroadPhaseMode` - 충돌 파이프라인
- IK + 물리 + 조작 계획 통합
- 그리퍼 제어

**실습:** 물체 집기 성공/실패 조건 분석

---

## 실습 체크리스트

### 기초
- [ ] IK로 로봇 팔 타겟 위치 제어
- [ ] IMU 센서 값 읽기
- [ ] 다중 월드 시뮬레이션 실행

### 중급
- [ ] 접촉력 센서로 힘 측정
- [ ] MJCF/USD 로봇 모델 로드
- [ ] 휴머노이드 시뮬레이션

### 고급
- [ ] 카메라 센서로 이미지 획득
- [ ] Pick-and-place 조작 작업
- [ ] Hydroelastic 접촉 시뮬레이션

---

## 각 예제 실행 명령어 모음

```bash
# Manipulator
uv run -m newton.examples ik_franka
uv run -m newton.examples robot_ur10
uv run -m newton.examples robot_allegro_hand
uv run -m newton.examples robot_panda_hydro

# Humanoid
uv run -m newton.examples robot_humanoid
uv run -m newton.examples robot_h1
uv run -m newton.examples robot_g1

# Sensors
uv run -m newton.examples sensor_imu
uv run -m newton.examples sensor_contact
uv run -m newton.examples sensor_tiled_camera
```

---

## 추천 학습 순서 (빠른 버전)

시간이 부족하다면 이 순서로 핵심만:

1. `ik_franka` → IK 기본
2. `robot_ur10` → 다중 월드 + 관절 제어
3. `sensor_contact` → 센서 기본
4. `robot_h1` → 실제 휴머노이드
5. `robot_panda_hydro` → 조작 작업 통합

---

## 내 로봇 모델 불러오기

Newton은 **URDF**, **MJCF**, **USD** 세 가지 포맷을 지원합니다.

### 지원 포맷 비교

| 포맷 | 확장자 | 특징 | 주로 사용 |
|------|--------|------|----------|
| URDF | `.urdf` | ROS 표준, 가장 범용적 | 산업용 로봇, ROS 연동 |
| MJCF | `.xml` | MuJoCo 네이티브, 물리 옵션 풍부 | RL 연구, MuJoCo 모델 |
| USD | `.usd`, `.usda` | NVIDIA 표준, 복잡한 씬 지원 | Isaac Sim, 대규모 씬 |

---

### 1. URDF 로드하기 (가장 일반적)

```python
import newton
import warp as wp

builder = newton.ModelBuilder()

# 로컬 URDF 파일 로드
builder.add_urdf(
    "/path/to/my_robot.urdf",
    xform=wp.transform(wp.vec3(0, 0, 1), wp.quat_identity()),  # 초기 위치
    floating=True,           # True: 자유 이동 / False: 바닥 고정
    scale=1.0,               # 스케일 조정
    enable_self_collisions=False,  # 자기 충돌 여부
    collapse_fixed_joints=True,    # 고정 조인트 병합 (성능 향상)
)

builder.add_ground_plane()
model = builder.finalize()
```

**주요 파라미터:**
- `floating`: 로봇이 공중에 떠있을 수 있는지 (휴머노이드는 True, 고정 팔은 False)
- `scale`: 모델 크기 조정
- `ignore_inertial_definitions`: True면 지오메트리에서 관성 계산 (기본값)

---

### 2. MJCF 로드하기 (MuJoCo 모델)

```python
builder = newton.ModelBuilder()
builder.default_shape_cfg.mu = 0.75  # 마찰 계수

builder.add_mjcf(
    "/path/to/my_robot.xml",
    xform=wp.transform(wp.vec3(0, 0, 1), wp.quat_identity()),
    ignore_names=["floor", "ground"],  # 제외할 바디 (정규식)
    parse_sites=True,          # 사이트(참조점) 로드
    parse_mujoco_options=True, # MuJoCo 옵션 파싱
)

model = builder.finalize()
```

**주요 파라미터:**
- `ignore_names`: 로드하지 않을 바디 이름 (정규식 패턴)
- `visual_classes`: 시각용 지오메트리 클래스
- `collider_classes`: 충돌용 지오메트리 클래스

---

### 3. USD 로드하기 (NVIDIA/Isaac)

```python
builder = newton.ModelBuilder()

# MuJoCo 솔버 사용 시 커스텀 속성 등록
newton.solvers.SolverMuJoCo.register_custom_attributes(builder)

builder.add_usd(
    "/path/to/my_robot.usda",
    ignore_paths=["/GroundPlane"],  # 제외할 경로
    collapse_fixed_joints=True,
    enable_self_collisions=False,
    hide_collision_shapes=True,    # 충돌 메시 숨기기
)

# 메시 단순화 (성능 향상)
builder.approximate_meshes("bounding_box")

model = builder.finalize()
```

**주요 파라미터:**
- `ignore_paths`: 로드하지 않을 USD 경로 (정규식)
- `approximate_meshes()`: 충돌 메시를 바운딩 박스/구/캡슐로 근사

---

### 에셋 가져오기

#### 방법 1: Newton 내장 에셋 사용
```python
# 내장 에셋 경로 가져오기
asset_path = newton.examples.get_asset("quadruped.urdf")
builder.add_urdf(asset_path, floating=True)
```

**내장 에셋 목록:**
- `quadruped.urdf`, `cartpole.urdf`
- `nv_ant.xml`, `nv_humanoid.xml`, `tabletop.xml`
- `cartpole.usda`, `ant.usda`, `humanoid.usda`, `bear.usd` 등

#### 방법 2: 원격 에셋 다운로드 (자동 캐시)
```python
# 원격 에셋 다운로드 (한 번만, 이후 캐시 사용)
asset_path = newton.utils.download_asset("unitree_h1")
usd_file = str(asset_path / "usd" / "h1_minimal.usda")
builder.add_usd(usd_file)
```

**다운로드 가능한 에셋:**
- `unitree_h1`, `unitree_g1` - 휴머노이드
- `franka_emika_panda` - 로봇 팔
- `anybotics_anymal_c`, `anybotics_anymal_d` - 4족 로봇
- `universal_robots_ur10` - 협동 로봇

---

### 실습: 내 로봇 시뮬레이션 템플릿

`playground/experiments/`에 저장하고 사용하세요:

```python
# playground/experiments/load_my_robot.py
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

        builder = newton.ModelBuilder()

        # ===== 여기에 로봇 로드 =====
        # 옵션 1: URDF
        # builder.add_urdf("/path/to/robot.urdf", floating=True)

        # 옵션 2: 내장 에셋
        builder.add_urdf(
            newton.examples.get_asset("quadruped.urdf"),
            xform=wp.transform(wp.vec3(0, 0, 0.5), wp.quat_identity()),
            floating=True,
        )
        # ===========================

        builder.add_ground_plane()
        self.model = builder.finalize()

        self.solver = newton.solvers.SolverXPBD(self.model)
        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()

        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.state_0)

        self.collision_pipeline = newton.examples.create_collision_pipeline(self.model, args)
        self.contacts = self.model.collide(self.state_0, collision_pipeline=self.collision_pipeline)

        self.viewer.set_model(self.model)

    def step(self):
        for _ in range(self.sim_substeps):
            self.state_0.clear_forces()
            self.contacts = self.model.collide(self.state_0, collision_pipeline=self.collision_pipeline)
            self.solver.step(self.state_0, self.state_1, self.control, self.contacts, self.sim_dt)
            self.state_0, self.state_1 = self.state_1, self.state_0
        self.sim_time += self.frame_dt

    def render(self):
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        self.viewer.end_frame()

if __name__ == "__main__":
    viewer, args = newton.examples.init()
    example = Example(viewer, args)
    newton.examples.run(example, args)
```

**실행:**
```bash
uv run python playground/experiments/load_my_robot.py
```

---

### 트러블슈팅

| 문제 | 해결 |
|------|------|
| 로봇이 바닥에 빠짐 | `floating=True` 또는 초기 z 위치 높이기 |
| 메시 로드 실패 | URDF와 같은 폴더에 mesh 파일 있는지 확인 |
| 너무 느림 | `approximate_meshes("bounding_box")` 사용 |
| 자기 충돌 문제 | `enable_self_collisions=False` |
| 스케일이 이상함 | `scale` 파라미터 조정 (미터 단위 확인) |

---

### 실습 체크리스트 (내 로봇)

- [ ] 내장 quadruped.urdf 로드해보기
- [ ] 원격 에셋 (unitree_h1) 다운로드 및 로드
- [ ] 내 URDF 파일 로드해보기
- [ ] floating vs 고정 차이 확인
- [ ] approximate_meshes 성능 차이 확인
