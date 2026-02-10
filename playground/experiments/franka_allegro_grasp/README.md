# Franka + Allegro Hand Cube Grasping

Franka Emika Panda 로봇 팔과 Allegro 손을 결합하여 테이블 위의 큐브를 잡아 들어올리는 강화학습 환경입니다.

[DEXTRAH](https://github.com/NVlabs/DEXTRAH)를 참고하여 구현되었습니다.

## Task 설명

**목표**: Franka arm + Allegro hand로 테이블 위의 큐브를 잡아서 목표 높이까지 들어올리기

### Task Phase (시각화용)

Phase는 시각화 및 로깅 목적으로 추적되지만, **보상은 DEXTRAH 원본처럼 continuous하게 계산**됩니다:

1. **Reach**: End effector가 큐브에 접근 (hand_to_object_dist < 0.3m)
2. **Grasp**: 손가락으로 큐브를 잡는 중 (cube_height > table + 0.05m)
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
- 카메라는 테이블 짧은 면(로봇 쪽)에 고정 설치
- 깊이값은 `[depth_min, depth_max]` 범위로 정규화 (0~1)

## Action Space

이 환경은 두 가지 액션 모드를 지원합니다:

### Direct Joint Control (기본, 23 dims)

`use_fabric_actions=False` (기본값)일 때:

| 항목 | 차원 | 설명 |
|------|------|------|
| Franka joint deltas | 7 | 팔 관절 위치 변화량 |
| Allegro joint deltas | 16 | 손가락 관절 위치 변화량 |

- **Delta position control**: 현재 위치에서의 변화량
- 범위: `[-1, 1] × action_scale` (기본 0.1)
- Joint limits로 클램핑

### FABRICS Action Space (DEXTRAH 스타일, 11 dims)

`use_fabric_actions=True`일 때 (DEXTRAH 원본과 동일):

| 항목 | 차원 | 설명 |
|------|------|------|
| Palm pose XYZ | 3 | End-effector 목표 위치 |
| Palm pose RPY | 3 | End-effector 목표 방향 (Roll-Pitch-Yaw) |
| Hand PCA | 5 | 손가락 PCA 좌표 (16D → 5D 차원 축소) |

- **Palm Pose Control**: Differential IK를 사용하여 EE 목표 위치/방향을 관절 위치로 변환
- **Hand PCA Control**: DEXTRAH 원본 PCA 행렬을 사용하여 5D PCA 좌표를 16D 관절 위치로 변환
- 범위: 정규화된 [-1, 1] 입력이 실제 작업 공간 범위로 매핑됨

```python
# FABRICS action 모드 활성화
config = EnvConfig(use_fabric_actions=True)
env = FrankaAllegroGraspEnv(config)
print(f"Action space: {env.num_actions}D")  # 11D
```

## Reward Structure

DEXTRAH 원본의 continuous reward 구조를 사용합니다. 모든 보상 컴포넌트가 동시에 계산되고 합산됩니다.

### DEXTRAH Reward Components (Continuous Sum)

```
total_reward = hand_to_object + object_to_goal + finger_curl_reg + lift + success_bonus
```

| Component | 수식 | Weight | Sharpness |
|-----------|------|--------|-----------|
| `hand_to_object` | `w × exp(-s × dist_ee_to_cube)` | 1.0 | 10.0 |
| `object_to_goal` | `w × exp(-s × dist_cube_to_goal)` | 5.0 | 15.0 |
| `lift` | `w × exp(-s × vertical_error)` | 5.0 | 8.5 |
| `finger_curl_reg` | `w × ‖q - curled_q‖²` | -0.01 | - |
| `success_bonus` | `w (if in_success_region)` | 10.0 | - |

### Thresholds (DEXTRAH 원본 값)

| 항목 | 값 | 설명 |
|------|-----|------|
| `hand_to_object_dist_threshold` | 0.3m | 도달 판정 임계값 |
| `object_goal_tol` | 0.1m | 성공 판정 거리 |
| `object_height_thresh` | 0.15m | 리프트 성공 높이 |
| `min_episode_steps` | 60 | 최소 에피소드 길이 |

## 실행 방법

### 시각화

```bash
# 기본 실행 (depth sensor, matplotlib 시각화, 카메라 디버그 모두 기본 활성화)
uv run --extra examples --extra torch-cu12 --with matplotlib python -m playground.experiments.franka_allegro_grasp.visualize

# 다중 환경
uv run --extra examples --extra torch-cu12 --with matplotlib python -m playground.experiments.franka_allegro_grasp.visualize --num-envs 16

# 특정 기능 비활성화
uv run --extra examples --extra torch-cu12 python -m playground.experiments.franka_allegro_grasp.visualize \
    --no-depth              # depth sensor 비활성화
    --no-show-depth         # matplotlib depth 창 숨기기
    --no-debug-camera       # 카메라 디버그 시각화 숨기기

# 랜덤 액션
uv run --extra examples --extra torch-cu12 --with matplotlib python -m playground.experiments.franka_allegro_grasp.visualize --random

# 학습된 정책 시각화
uv run --extra examples --extra torch-cu12 --with matplotlib python -m playground.experiments.franka_allegro_grasp.visualize \
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

### Simulation (DEXTRAH 원본)

```python
fps: int = 120                # Physics at 120Hz (DEXTRAH: sim_dt = 1/120)
sim_substeps: int = 1         # No additional substeps
control_decimation: int = 2   # 60Hz control (DEXTRAH: decimation = 2)
episode_length: int = 600     # 10 seconds (DEXTRAH: episode_length_s = 10.0)
min_episode_steps: int = 60   # DEXTRAH minimum
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

# Goal (DEXTRAH values)
lift_height: float = 0.15            # lift 15cm above table (object_height_thresh)
goal_tolerance: float = 0.1          # success distance tolerance (object_goal_tol)
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
- 위치: 각 환경의 (-0.3, 0.1, 0.43) - 로봇 옆, 낮은 높이
- 방향: 큐브 영역 (-0.3, -0.5, 0.45) 방향

## 구현 세부사항

### Newton SensorTiledCamera

Depth sensor는 Newton의 `SensorTiledCamera`를 사용합니다:

```python
from newton.sensors import SensorTiledCamera

# 센서 생성 (new API: width/height/num_cameras는 create_*_output 메서드로 이동)
depth_sensor = SensorTiledCamera(
    model=model,
    options=SensorTiledCamera.Options(
        default_light=True,
        colors_per_shape=True,
    ),
)

# 카메라 ray 계산 (new API: width, height 파라미터 추가)
camera_rays = depth_sensor.compute_pinhole_camera_rays(
    width=160, height=120, fov=fov_radians
)

# 깊이 이미지 버퍼 생성 (new API: width, height, num_cameras 파라미터)
# 출력 shape: (num_worlds, num_cameras, height, width)
depth_image = depth_sensor.create_depth_image_output(
    width=160, height=120, num_cameras=1
)

# 렌더링
depth_sensor.render(
    state=state,
    camera_transforms=camera_transforms,
    camera_rays=camera_rays,
    depth_image=depth_image,
)
```

### Phase Transition (시각화용)

Task phase는 시각화 목적으로 자동 전환됩니다. **보상은 phase와 무관하게 continuous하게 계산**됩니다:

```
Reach → Grasp: EE-큐브 거리 < hand_to_object_dist_threshold (0.3m)
Grasp → Lift: 큐브 높이 > table_height + 0.05m
```

**중요**: DEXTRAH 원본과 동일하게, 모든 reward component는 항상 동시에 계산되고 합산됩니다.

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

이 예제는 [FABRICS](https://github.com/NVlabs/FABRICS) (Riemannian Geometric Fabrics)와 [DEXTRAH](https://github.com/NVlabs/DEXTRAH) 스타일의 파지 보상 및 액션 공간을 통합합니다.

### FabricActionController (DEXTRAH Action Space)

`FabricActionController`는 DEXTRAH 원본과 동일한 11D 액션 공간을 구현합니다:

#### 액션 공간 구조

| 인덱스 | 항목 | 차원 | 범위 (정규화 후) | 설명 |
|--------|------|------|-----------------|------|
| 0-2 | Palm XYZ | 3 | [-0.6, 0.0] × [-0.9, -0.1] × [0.35, 0.8] | EE 목표 위치 (m) |
| 3-5 | Palm RPY | 3 | [-π, π] × [-π/4, π/4] × [-π, π] | EE 목표 방향 (rad) |
| 6-10 | Hand PCA | 5 | [0.25, 3.83] × [-0.33, 3.00] × ... | PCA 좌표 |

#### 변환 파이프라인

```
11D Fabric Action ([-1, 1])
    ↓
Normalize to actual ranges
    ↓
┌─────────────────────────────────────────────┐
│ 6D Palm Pose          │ 5D Hand PCA         │
│ (XYZ + RPY)           │                     │
└─────────────────────────────────────────────┘
    ↓                           ↓
Differential IK             PCA Projection
(Damped Least Squares)      (5×16 Matrix)
    ↓                           ↓
7D Franka Joints        16D Allegro Joints
    ↓                           ↓
└──────────── 23D Joint Targets ─────────────┘
```

#### Hand PCA Matrix (DEXTRAH/FABRICS 원본)

5×16 PCA 행렬로 5D 저차원 공간에서 16D Allegro 관절을 제어합니다:

```python
# fabric.py에 정의된 원본 PCA 행렬
HAND_PCA_MATRIX = [
    # PC1: Power grasp (모든 손가락 동시 굽힘)
    [-0.039, 0.379, 0.447, 0.007, 0.002, 0.320, 0.447, 0.052, ...],
    # PC2: Thumb-index precision (엄지+검지)
    [-0.051, -0.130, 0.058, 0.579, 0.010, -0.185, 0.054, 0.549, ...],
    # PC3: Index extension (검지 펴기)
    [-0.057, -0.347, 0.334, -0.180, -0.044, -0.477, 0.325, -0.152, ...],
    # PC4: Thumb opposition (엄지 대립)
    [0.023, -0.034, 0.034, -0.027, 0.023, 0.046, 0.098, -0.001, ...],
    # PC5: Finger spread (손가락 벌리기)
    [-0.045, -0.472, 0.093, 0.231, -0.002, 0.096, 0.125, 0.037, ...],
]
```

#### Hand PCA 좌표 범위 (DEXTRAH 원본)

| PCA | Min | Max | 설명 |
|-----|-----|-----|------|
| PC1 | 0.2475 | 3.8336 | Power grasp 강도 |
| PC2 | -0.3286 | 3.0025 | Precision grasp 강도 |
| PC3 | -0.7238 | 0.8977 | Index extension |
| PC4 | -0.0192 | 1.0243 | Thumb opposition |
| PC5 | -0.5532 | 0.0629 | Finger spread |

#### 코드 예시

```python
from playground.experiments.franka_allegro_grasp import (
    FabricActionController,
    FrankaAllegroGraspEnv,
    EnvConfig,
    HAND_PCA_MATRIX,  # 5×16 PCA 행렬
    HAND_PCA_MINS,    # [0.2475, -0.3286, ...]
    HAND_PCA_MAXS,    # [3.8336, 3.0025, ...]
)
import torch

# 1. FabricActionController 직접 사용
controller = FabricActionController(
    franka_dof=7,
    allegro_dof=16,
    device="cuda",
    damping=0.1,       # IK 댐핑
    ik_step_size=0.5,  # IK 스텝 크기
)

# 11D fabric actions → 23D joint targets
fabric_action = torch.randn(batch_size, 11)  # [-1, 1]
franka_target, allegro_target = controller(
    fabric_action,
    current_franka_q,
    current_ee_pos,
    current_ee_quat,
)

# 2. 환경에서 직접 사용
config = EnvConfig(
    num_envs=256,
    use_fabric_actions=True,  # 11D 액션 공간 활성화
    fabric_ik_damping=0.1,
    fabric_ik_step_size=0.5,
    fabric_decimation=2,
)
env = FrankaAllegroGraspEnv(config)
print(f"Action dim: {env.num_actions}")  # 11

obs = env.reset()
action = torch.randn(256, 11, device="cuda").clamp(-1, 1)
obs, reward, done, info = env.step(action)
```

### 환경 통합

`FrankaAllegroGraspEnv`는 내부적으로 두 가지 FABRICS 모듈을 사용:

1. **`FabricActionController`** (`use_fabric_actions=True` 시):
   - 11D fabric actions를 23D joint targets로 변환
   - DEXTRAH 원본과 동일한 액션 공간 사용
   - fabric_decimation 횟수만큼 IK 반복 실행

2. **`GraspFabric`** (항상 활성화):
   - 매 스텝마다 grasp features 계산
   - FABRICS 기반 보상을 reward에 직접 통합
   - 손가락 위치 기반 파지 품질 평가

### FABRICS 설정 파라미터

```python
# config.py의 FABRICS 관련 설정
use_fabric_actions: bool = False   # True: 11D, False: 23D
fabric_ik_damping: float = 0.1     # Differential IK 댐핑 계수
fabric_ik_step_size: float = 0.5   # IK 업데이트 스텝 크기
fabric_decimation: int = 2         # 물리 스텝당 fabric 스텝 수
```

### FABRICS 핵심 개념

1. **Task Map**: 관절 공간(joint space)을 작업 공간(task space)으로 매핑
   - `LinearTaskMap`: PCA 기반 손 제어 (synergy-based control)
   - `FingertipTaskMap`: 손가락 끝 위치 계산 (forward kinematics)

2. **Grasp Features**: 파지 관련 특징 계산
   - `fingertip_positions`: 월드 좌표계 손가락 끝 위치 (batch, 4, 3)
   - `fingertip_distances`: 손가락 끝-큐브 거리 (batch, 4)
   - `grasp_closure`: 손가락 중심점과 큐브 중심 거리 (batch,)
   - `palm_to_cube`: Palm-to-cube 벡터 (batch, 3)

3. **Grasp Rewards**: 파지 기반 보상 (env.py에 직접 통합)
   - `contact_reward`: 손가락이 큐브 표면에 닿을 때
   - `closure_reward`: 손가락이 큐브를 감쌀 때
   - `approach_reward`: 손바닥이 큐브에 가까울 때
   - `multi_contact_bonus`: 여러 손가락이 동시에 접촉할 때

### Training Pipeline

PPO 학습 시 FABRICS 액션/관측을 사용:
- `--fabric-actions` 옵션으로 11D 액션 공간 사용
- `--no-fabric` 옵션으로 fabric features 비활성화

### GraspFabric 코드 예시

```python
from playground.experiments.franka_allegro_grasp import GraspFabric

fabric = GraspFabric(franka_dof=7, allegro_dof=16, device="cuda")

# Compute grasp features
grasp_features = fabric.compute_grasp_features(
    ee_pos, ee_quat, allegro_q, cube_pos, cube_size=0.05
)

# Get grasp rewards
grasp_rewards = fabric.compute_grasp_reward(grasp_features, cube_size=0.05)

# Access individual components
contact_reward = grasp_rewards["contact_reward"]      # (batch,)
closure_reward = grasp_rewards["closure_reward"]      # (batch,)
multi_contact = grasp_rewards["multi_contact_bonus"]  # (batch,)
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
# Full training with wandb (DEXTRAH default values)
uv run --extra examples --extra torch-cu12 python -m playground.experiments.franka_allegro_grasp.train \
    --num-envs 256 \
    --total-timesteps 100000000 \
    --learning-rate 5e-4 \
    --rollout-steps 16 \
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
├── config.py        # 환경/PPO 설정 (use_fabric_actions 포함)
├── env.py           # FrankaAllegroGraspEnv (두 가지 액션 모드 지원)
├── fabric.py        # FABRICS 구현:
│   │                #   - FabricActionController: 11D→23D 변환
│   │                #   - GraspFabric: 파지 특징/보상 계산
│   │                #   - HAND_PCA_MATRIX: DEXTRAH 원본 PCA 행렬
│   │                #   - TaskMap, LinearTaskMap, FingertipTaskMap
├── train.py         # PPO 학습 스크립트
├── visualize.py     # Newton viewer 시각화 (+ matplotlib depth)
├── view_depth.py    # Depth 이미지 확인 스크립트
└── README.md
```

## PPO 하이퍼파라미터 (DEXTRAH 원본)

```python
# Learning (DEXTRAH values)
learning_rate: float = 5e-4         # DEXTRAH: 5e-4
lr_schedule: str = "adaptive"       # Adaptive LR based on KL
gamma: float = 0.99
gae_lambda: float = 0.95            # DEXTRAH: tau = 0.95

# Clipping
clip_epsilon: float = 0.2           # DEXTRAH: e_clip = 0.2
kl_threshold: float = 0.016         # DEXTRAH: adaptive LR threshold

# Loss coefficients
entropy_coef: float = 0.0           # DEXTRAH: 0.0
value_coef: float = 4.0             # DEXTRAH: critic_coef = 4
bounds_loss_coef: float = 0.0001    # DEXTRAH: action bounds penalty
max_grad_norm: float = 1.0

# Training dynamics
num_epochs: int = 5                 # DEXTRAH: mini_epochs = 5
rollout_steps: int = 16             # DEXTRAH: horizon_length = 16
minibatch_size: int = 8192          # DEXTRAH: 8192

# Normalization
normalize_input: bool = True
normalize_value: bool = True        # RunningMeanStd로 value target 정규화
normalize_advantage: bool = True
observation_clip: float = 5.0
action_clip: float = 1.0

# Network (DEXTRAH: [512, 512, 256, 128])
hidden_dims: tuple = (512, 512, 256, 128)
activation: str = "elu"
```

### PPO 알고리즘 (rl_games 원본 정렬)

PPO 알고리즘은 rl_games 라이브러리(DextrEme/DEXTRAH에서 사용)와 동일하게 구현되어 있습니다:

| 항목 | rl_games 원본 | 이 구현 |
|------|-------------|--------|
| KL divergence | Analytical Gaussian KL | Analytical Gaussian KL ✓ |
| KL early stopping | 사용하지 않음 | 사용하지 않음 ✓ |
| Adaptive LR | per epoch, /1.5 decrease, ×1.5 increase | 동일 ✓ |
| Value loss | Clipped (max of unclipped, clipped) | Clipped ✓ |
| Bounds loss | Action mean (μ), soft bound 1.1 | Action mean (μ), soft bound 1.1 ✓ |
| Value normalization | RunningMeanStd | RunningMeanStd ✓ |
| log_std clamp | [-5.0, 2.0] | [-5.0, 2.0] ✓ |

#### Value Normalization (RunningMeanStd)

`normalize_value: True` 설정 시 RunningMeanStd를 사용하여 value target을 정규화합니다:

```
Critic output (normalized space)
    → denormalize for GAE computation (real reward scale)
    → compute returns and advantages
    → update normalizer statistics with returns
    → normalize returns/old_values for value loss (back to normalized space)
```

이를 통해 value_loss의 스케일이 안정화되어 학습 중 들쑥날쑥한 value_loss를 방지합니다.

## DEXTRAH 원본 정렬

이 구현은 [NVlabs/DEXTRAH](https://github.com/NVlabs/DEXTRAH)의 원본 값들과 정렬되어 있습니다:

### Simulation

| 항목 | DEXTRAH 원본 | 이 구현 |
|------|-------------|--------|
| Physics frequency | 120 Hz | 120 Hz ✓ |
| Control frequency | 60 Hz | 60 Hz ✓ |
| Episode length | 10.0s | 10.0s ✓ |
| Min episode steps | 60 | 60 ✓ |

### Reward Structure

| 항목 | DEXTRAH 원본 | 이 구현 |
|------|-------------|--------|
| hand_to_object weight | 1.0 | 1.0 ✓ |
| hand_to_object sharpness | 10.0 | 10.0 ✓ |
| object_to_goal weight | 5.0 | 5.0 ✓ |
| lift weight | 5.0 | 5.0 ✓ |
| lift sharpness | 8.5 | 8.5 ✓ |
| finger_curl_reg weight | -0.01 | -0.01 ✓ |
| success bonus | 10.0 | 10.0 ✓ |

### Thresholds

| 항목 | DEXTRAH 원본 | 이 구현 |
|------|-------------|--------|
| hand_to_object_dist_threshold | 0.3m | 0.3m ✓ |
| object_goal_tol | 0.1m | 0.1m ✓ |
| object_height_thresh | 0.15m | 0.15m ✓ |

### PPO Hyperparameters

| 항목 | DEXTRAH 원본 | 이 구현 |
|------|-------------|--------|
| learning_rate | 5e-4 | 5e-4 ✓ |
| lr_schedule | adaptive | adaptive ✓ |
| gamma | 0.99 | 0.99 ✓ |
| tau (gae_lambda) | 0.95 | 0.95 ✓ |
| e_clip | 0.2 | 0.2 ✓ |
| kl_threshold | 0.016 | 0.016 ✓ |
| entropy_coef | 0.0 | 0.0 ✓ |
| critic_coef (value_coef) | 4.0 | 4.0 ✓ |
| bounds_loss_coef | 0.0001 | 0.0001 ✓ |
| horizon_length | 16 | 16 ✓ |
| mini_epochs | 5 | 5 ✓ |
| network | [512,512,256,128] | [512,512,256,128] ✓ |
| activation | elu | elu ✓ |
| normalize_value | True | True (RunningMeanStd) ✓ |
| normalize_input | True | True ✓ |
| normalize_advantage | True | True ✓ |

### PPO Algorithm (rl_games)

| 항목 | rl_games 원본 | 이 구현 |
|------|-------------|--------|
| KL divergence | Analytical Gaussian | Analytical Gaussian ✓ |
| KL early stopping | None | None ✓ |
| Adaptive LR timing | Per epoch | Per epoch ✓ |
| LR decrease factor | /1.5 | /1.5 ✓ |
| LR increase factor | ×1.5 | ×1.5 ✓ |
| Value loss | Clipped | Clipped ✓ |
| Bounds loss target | Action mean (μ) | Action mean (μ) ✓ |
| Bounds loss soft bound | 1.1 | 1.1 ✓ |
| Value normalization | RunningMeanStd | RunningMeanStd ✓ |
| log_std clamp | [-5.0, 2.0] | [-5.0, 2.0] ✓ |

### Action Space (FABRICS)

| 항목 | DEXTRAH 원본 | 이 구현 |
|------|-------------|--------|
| Action dimensions | 11 (6 palm + 5 PCA) | 11 (use_fabric_actions=True) ✓ |
| Palm pose | 6D (XYZ + RPY) | 6D ✓ |
| Hand PCA | 5D | 5D ✓ |
| PCA matrix | 5×16 from FABRICS | 5×16 원본 사용 ✓ |
| Hand PCA mins | [0.25, -0.33, -0.72, -0.02, -0.55] | 동일 ✓ |
| Hand PCA maxs | [3.83, 3.00, 0.90, 1.02, 0.06] | 동일 ✓ |
| Fabric decimation | 2 | 2 ✓ |

### 주요 차이점

- **로봇**: DEXTRAH는 Kuka 팔 사용, 이 구현은 Franka Emika Panda 사용
- **시뮬레이터**: DEXTRAH는 Isaac Lab 사용, 이 구현은 Newton 사용
- **IK 방식**: DEXTRAH는 FABRICS geometric IK, 이 구현은 Differential IK (Damped Least Squares)
- **Palm Pose 범위**: Franka 작업 공간에 맞게 조정됨
- **FABRICS**: 학습 관측에 추가적으로 FABRICS features 사용 가능 (`--no-fabric`으로 비활성화)

## 참고 자료

- [FABRICS](https://github.com/NVlabs/FABRICS) - Riemannian Geometric Fabrics
- [DEXTRAH](https://github.com/NVlabs/DEXTRAH) - Kuka + Allegro grasping (원본 참조)
- [Newton SensorTiledCamera](../../newton/sensors.py) - Depth sensor API
- [Franka Allegro Example](../franka_allegro/) - 기본 결합 예제
- [CleanRL](https://github.com/vwxyzjn/cleanrl) - PPO 구현 참고
