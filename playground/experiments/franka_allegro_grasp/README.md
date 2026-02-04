# Franka + Allegro Hand Cube Grasping

Franka Emika Panda 로봇 팔과 Allegro 손을 결합하여 테이블 위의 큐브를 잡아 들어올리는 강화학습 환경입니다.

[DEXTRAH](https://github.com/NVlabs/DEXTRAH)를 참고하여 구현되었습니다.

## Task 설명

**목표**: Franka arm + Allegro hand로 테이블 위의 큐브를 잡아서 목표 높이까지 들어올리기

### 3단계 Task Phase

1. **Reach**: End effector를 큐브 위치로 이동
2. **Grasp**: 손가락으로 큐브를 잡기
3. **Lift**: 큐브를 목표 높이까지 들어올리기

## Scene Layout

로봇, 테이블, 큐브의 배치는 로봇 팔의 자연스러운 도달 범위를 고려하여 설정되었습니다.

```
        Y
        ↑
        |
   +---------+
   |  Table  |  ← 테이블 중심: (-0.3, -0.5)
   |  [Cube] |  ← 큐브: (-0.3, -0.5, 0.45)
   +---------+
        |
      (Robot)   ← 로봇 베이스: (0.0, 0.25, 0.0)
        |           EE 위치: 약 (-0.37, -0.51)
        +------→ X
```

### 좌표 설정

| 항목 | 위치 | 설명 |
|------|------|------|
| Robot base | (0.0, 0.25, 0.0) | 테이블 Y축 중심에 맞춤 |
| Table center | (-0.3, -0.5) | 로봇 팔 도달 범위 내 |
| Table extends | X: [-0.6, 0.0], Y: [-0.9, -0.1] | 0.6m × 0.8m |
| Cube spawn | (-0.3, -0.5, 0.45) | 테이블 중심, 높이 0.45m |
| EE initial | (~-0.37, ~-0.51, ~0.66) | 큐브에서 약 5cm 거리 |

### 다중 환경 (Vectorized)

환경은 Grid 형태로 복제됩니다:
- **Spacing**: (1.5m, 1.5m, 0.0m)
- **Grid 순서**: `grid_x = env_idx // grid_size`, `grid_y = env_idx % grid_size`

```
Env 2 (1.5, 0)    Env 3 (1.5, 1.5)
Env 0 (0, 0)      Env 1 (0, 1.5)
```

## Observation Space

### State Observations (72 dims)

| 항목 | 차원 | 설명 |
|------|------|------|
| Franka joint positions | 7 | 팔 관절 위치 (normalized) |
| Franka joint velocities | 7 | 팔 관절 속도 (×0.1) |
| Allegro joint positions | 16 | 손가락 관절 위치 (normalized) |
| Allegro joint velocities | 16 | 손가락 관절 속도 (×0.1) |
| End effector position | 3 | EE 월드 좌표 |
| End effector orientation | 4 | EE 쿼터니언 |
| Cube relative position | 3 | EE 기준 큐브 상대 위치 |
| Cube orientation | 4 | 큐브 쿼터니언 |
| Cube linear velocity | 3 | 큐브 선속도 (×0.1) |
| Cube angular velocity | 3 | 큐브 각속도 (×0.1) |
| Goal position | 3 | 목표 위치 |
| Task phase | 3 | One-hot [reach, grasp, lift] |

### Depth Observations (Optional)

- **Depth image**: `depth_width × depth_height` (기본 160×120 = 19200 dims)
- 카메라는 End Effector에 부착
- 깊이값은 `[depth_min, depth_max]` 범위로 정규화

## Action Space (23 dims)

| 항목 | 차원 | 설명 |
|------|------|------|
| Franka joint deltas | 7 | 팔 관절 위치 변화량 |
| Allegro joint deltas | 16 | 손가락 관절 위치 변화량 |

- **Delta position control**: 현재 위치에서의 변화량
- 범위: `[-1, 1] × action_scale` (기본 0.1)
- Joint limits로 클램핑

## Reward Structure

### Phase별 Reward

| Phase | Reward | 설명 |
|-------|--------|------|
| Reach | `exp(-5 × dist_ee_to_cube)` | EE가 큐브에 가까워질수록 보상 |
| Grasp | `exp(-10 × dist_ee_to_cube)` | 큐브를 잡은 상태 유지 |
| Lift | `exp(-5 × dist_cube_to_goal)` | 큐브가 목표 높이에 가까워질수록 보상 |

### Bonus & Penalty

| 항목 | 값 | 조건 |
|------|-----|------|
| Reach bonus | +50 | EE-큐브 거리 < 0.1m |
| Grasp bonus | +100 | 큐브 잡기 성공 |
| Lift bonus | +250 | 목표 높이 도달 |
| Drop penalty | -100 | 큐브 낙하 |
| Action penalty | -0.0001 × Σ(a²) | 액션 크기 |

## 실행 방법

### 시각화

```bash
# 기본 실행 (스크립트 데모)
uv run --extra examples --extra torch-cu12 python -m playground.experiments.franka_allegro_grasp.visualize

# 다중 환경
uv run --extra examples --extra torch-cu12 python -m playground.experiments.franka_allegro_grasp.visualize --num-envs 16

# Depth sensor 활성화
uv run --extra examples --extra torch-cu12 python -m playground.experiments.franka_allegro_grasp.visualize --use-depth

# Depth sensor + 실시간 matplotlib 시각화 (Newton 뷰어와 함께)
uv run --extra examples --extra torch-cu12 --with matplotlib python -m playground.experiments.franka_allegro_grasp.visualize --use-depth --show-depth

# 카메라 위치 및 FOV 디버그 시각화 (주황색 구: 카메라 위치, 청록색: FOV)
uv run --extra examples --extra torch-cu12 python -m playground.experiments.franka_allegro_grasp.visualize --use-depth --debug-camera

# 랜덤 액션
uv run --extra examples --extra torch-cu12 python -m playground.experiments.franka_allegro_grasp.visualize --random

# 학습된 정책 시각화
uv run --extra examples --extra torch-cu12 python -m playground.experiments.franka_allegro_grasp.visualize \
    --checkpoint checkpoints/franka_allegro_grasp_XXXXXX/final.pt
```

### Depth 이미지 확인

```bash
# 실시간 matplotlib 창으로 depth 이미지 확인 (별도 실행)
uv run --extra examples --extra torch-cu12 --with matplotlib python -m playground.experiments.franka_allegro_grasp.view_depth --realtime

# PNG 파일로 저장
uv run --extra examples --extra torch-cu12 python -m playground.experiments.franka_allegro_grasp.view_depth --save
```

## 환경 설정

### Simulation

```python
fps: int = 60
sim_substeps: int = 4
control_decimation: int = 2  # 30Hz control
episode_length: int = 400    # ~6.7 seconds
```

### Robot Parameters

```python
# Franka Arm
franka_stiffness: float = 500.0
franka_damping: float = 100.0
franka_effort_limit: float = 80.0
franka_armature: float = 0.1

# Allegro Hand
hand_stiffness: float = 200.0
hand_damping: float = 20.0
hand_effort_limit: float = 20.0
hand_armature: float = 0.05
```

### Scene

```python
# Table - positioned in front of robot arm
table_height: float = 0.4
table_size: tuple = (0.6, 0.8, 0.02)  # width, depth, thickness
table_pos: tuple = (-0.3, -0.5)       # center position

# Cube
cube_size: float = 0.05              # 5cm cube
cube_mass: float = 0.1               # 100g
cube_spawn_pos: tuple = (-0.3, -0.5, 0.45)  # on table center
cube_spawn_noise: float = 0.05       # ±5cm randomization

# Goal
lift_height: float = 0.2             # lift 20cm above table
```

### Depth Sensor

```python
use_depth_sensor: bool = True
depth_width: int = 160
depth_height: int = 120
depth_fov: float = 60.0      # degrees
depth_min: float = 0.1       # meters
depth_max: float = 2.0       # meters
```

**카메라 위치**: 테이블 짧은 면(로봇 쪽)에서 테이블/큐브를 바라보는 고정 뷰
- 위치: 각 환경의 (-0.3, 0.1, 0.55) - 로봇 옆, 낮은 높이
- 방향: 큐브 영역 (-0.3, -0.5, 0.45) 방향

## 구현 세부사항

### Newton SensorTiledCamera

Depth sensor는 Newton의 `SensorTiledCamera`를 사용합니다:

```python
from newton.sensors import SensorTiledCamera

# 센서 생성
depth_sensor = SensorTiledCamera(
    model=model,
    num_cameras=1,
    width=160,
    height=120,
    options=SensorTiledCamera.Options(
        default_light=True,
        colors_per_shape=True,
    ),
)

# 카메라 ray 계산
camera_rays = depth_sensor.compute_pinhole_camera_rays(fov_radians)

# 깊이 이미지 버퍼 생성
depth_image = depth_sensor.create_depth_image_output()

# 렌더링
depth_sensor.render(
    state=state,
    camera_transforms=camera_transforms,
    camera_rays=camera_rays,
    depth_image=depth_image,
)
```

### Phase Transition

Task phase는 자동으로 전환됩니다:

```
Reach → Grasp: EE-큐브 거리 < reach_threshold (0.1m)
Grasp → Lift: 큐브 높이 > table_height + 0.05m
```

### Environment Replication

정적 물체(테이블)와 동적 물체(큐브)는 다르게 처리됩니다:

```python
# 1. Single environment 구성
single_env_builder = newton.ModelBuilder()
# Robot (Franka + Allegro) 추가
# Cube 추가 (free-floating body)

# 2. Replicate for multiple environments
builder.replicate(single_env_builder, num_envs, spacing=(1.5, 1.5, 0.0))

# 3. Static shapes (tables) are added AFTER replication
# (static shapes on body=-1 are not replicated automatically)
for env_idx in range(num_envs):
    table_pos = calculate_table_pos_for_env(env_idx)
    builder.add_shape_box(body=-1, xform=table_xform, ...)

# 4. Cube positions are corrected in reset()
# (explicitly set world coordinates based on grid layout)
```

## FABRICS 통합

이 예제는 [FABRICS](https://github.com/NVlabs/FABRICS) (Riemannian Geometric Fabrics)의 핵심 개념을 단순화하여 구현합니다.

### FABRICS 핵심 개념

1. **Task Map**: 관절 공간(joint space)을 작업 공간(task space)으로 매핑
   - `LinearTaskMap`: PCA 기반 손 제어
   - `FingertipTaskMap`: 손가락 끝 위치 계산

2. **Grasp Features**: 파지 관련 특징 계산
   - 손가락 끝-큐브 거리
   - Grasp closure (손가락 중심점과 큐브 중심 거리)
   - Palm-to-cube 벡터

3. **Grasp Rewards**: 파지 기반 보상
   - Contact reward: 손가락이 큐브 표면에 닿을 때
   - Closure reward: 손가락이 큐브를 감쌀 때
   - Multi-contact bonus: 여러 손가락이 동시에 접촉할 때

### 사용 예시

```python
from playground.experiments.franka_allegro_grasp import GraspFabric

fabric = GraspFabric(franka_dof=7, allegro_dof=16, device="cuda")

# Compute grasp features
grasp_features = fabric.compute_grasp_features(
    ee_pos, ee_quat, allegro_q, cube_pos, cube_size=0.05
)

# Get grasp rewards
grasp_rewards = fabric.compute_grasp_reward(grasp_features, cube_size=0.05)
```

## 학습

### 기본 학습

```bash
# 기본 설정 (256 envs, FABRICS 활성화)
uv run --extra examples --extra torch-cu12 python -m playground.experiments.franka_allegro_grasp.train

# 빠른 테스트
uv run --extra examples --extra torch-cu12 python -m playground.experiments.franka_allegro_grasp.train \
    --num-envs 64 \
    --total-timesteps 100000 \
    --no-wandb
```

### 학습 옵션

```bash
# Full training with wandb
uv run --extra examples --extra torch-cu12 python -m playground.experiments.franka_allegro_grasp.train \
    --num-envs 512 \
    --total-timesteps 50000000 \
    --learning-rate 3e-4 \
    --rollout-steps 24 \
    --wandb-project my-grasp-project

# Without FABRICS observations
uv run --extra examples --extra torch-cu12 python -m playground.experiments.franka_allegro_grasp.train \
    --no-fabric

# Resume from checkpoint
uv run --extra examples --extra torch-cu12 python -m playground.experiments.franka_allegro_grasp.train \
    --checkpoint checkpoints/franka_allegro_grasp_XXXXXX/checkpoint_1000000.pt
```

### 로깅

```bash
# TensorBoard
tensorboard --logdir runs/

# Weights & Biases (기본 활성화)
# --no-wandb 옵션으로 비활성화 가능
```

## 프로젝트 구조

```
franka_allegro_grasp/
├── __init__.py      # 패키지 export
├── config.py        # 환경/PPO 설정
├── env.py           # FrankaAllegroGraspEnv
├── fabric.py        # FABRICS 구현 (TaskMap, GraspFabric)
├── train.py         # PPO 학습 스크립트
├── visualize.py     # Newton viewer 시각화 (+ matplotlib depth)
├── view_depth.py    # Depth 이미지 확인 스크립트
└── README.md
```

## PPO 하이퍼파라미터

```python
learning_rate: float = 3e-4
gamma: float = 0.99
gae_lambda: float = 0.95
clip_epsilon: float = 0.2
entropy_coef: float = 0.01
value_coef: float = 0.5
max_grad_norm: float = 1.0
num_epochs: int = 5
num_minibatches: int = 4
rollout_steps: int = 24
hidden_dims: tuple = (512, 256, 128)
```

## 참고 자료

- [FABRICS](https://github.com/NVlabs/FABRICS) - Riemannian Geometric Fabrics
- [DEXTRAH](https://github.com/NVlabs/DEXTRAH) - Kuka + Allegro grasping
- [Newton SensorTiledCamera](../../newton/sensors.py) - Depth sensor API
- [Franka Allegro Example](../franka_allegro/) - 기본 결합 예제
- [CleanRL](https://github.com/vwxyzjn/cleanrl) - PPO 구현 참고
