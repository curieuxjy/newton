# Franka + Allegro Hand Cube Grasping (DEXTRAH-G)

Franka Emika Panda 로봇 팔과 Allegro 손을 결합하여 테이블 위의 큐브를 잡아 들어올리는 강화학습 환경입니다.

[DEXTRAH-G](https://github.com/NVlabs/DEXTRAH)의 2단계 teacher-student distillation 파이프라인을 구현합니다.

## 알고리즘 개요

DEXTRAH-G는 2단계 학습 파이프라인을 사용합니다:

```
Stage 1: Teacher (Privileged RL)          Stage 2: Student Distillation
┌──────────────────────────────┐          ┌─────────────────────────────────┐
│ Full state (172D)            │          │ Depth image (120×160)           │
│ ↓                            │          │ + Proprioception (159D)         │
│ LSTM(1024) Actor             │  ──→     │ ↓                               │
│ LSTM(2048) Critic            │ freeze   │ CNN → LSTM(512) → MLP           │
│ ↓                            │  teacher │ ↓                               │
│ 11D FABRICS action           │          │ L_action + β·L_pos              │
│ PPO + privileged object info │          │ Imitate teacher + predict obj   │
└──────────────────────────────┘          └─────────────────────────────────┘
```

**Stage 1 - Teacher**: 큐브의 위치/속도 등 privileged 정보를 사용하여 LSTM 기반 PPO로 teacher 정책을 학습합니다. Depth 이미지를 사용하지 않습니다.

**Stage 2 - Student Distillation**: Frozen teacher의 action을 모방하는 student 정책을 학습합니다. Student는 depth 이미지 + proprioception만 사용하며, auxiliary task로 object position을 예측합니다.

## 실행 방법

### Stage 1: Teacher 학습

```bash
# 기본 설정 (256 envs, privileged state, no depth)
uv run --extra examples --extra torch-cu12 python -m playground.experiments.franka_allegro_grasp.train

# 빠른 테스트
uv run --extra examples --extra torch-cu12 python -m playground.experiments.franka_allegro_grasp.train \
    --num-envs 64 --max-iterations 1000 --no-wandb

# Resume from checkpoint
uv run --extra examples --extra torch-cu12 python -m playground.experiments.franka_allegro_grasp.train \
    --checkpoint checkpoints/teacher_*/best.pt
```

### Stage 2: Student Distillation

```bash
# Teacher checkpoint를 사용하여 student 학습
uv run --extra examples --extra torch-cu12 python -m playground.experiments.franka_allegro_grasp.distill \
    --teacher-checkpoint checkpoints/teacher_*/best.pt

# 빠른 테스트
uv run --extra examples --extra torch-cu12 python -m playground.experiments.franka_allegro_grasp.distill \
    --teacher-checkpoint checkpoints/teacher_*/best.pt \
    --num-envs 64 --max-iterations 10000 --no-wandb
```

### 시각화

```bash
# 기본 실행 (depth sensor, matplotlib 시각화, 카메라 디버그 모두 기본 활성화)
uv run --extra examples --extra torch-cu12 --with matplotlib \
    python -m playground.experiments.franka_allegro_grasp.visualize

# 학습된 teacher 정책 시각화
uv run --extra examples --extra torch-cu12 --with matplotlib \
    python -m playground.experiments.franka_allegro_grasp.visualize \
    --checkpoint checkpoints/teacher_*/best.pt

# 학습된 student 정책 시각화 (depth 필요)
uv run --extra examples --extra torch-cu12 --with matplotlib \
    python -m playground.experiments.franka_allegro_grasp.visualize \
    --checkpoint checkpoints/student_*/best.pt

# 다중 환경 / 랜덤 액션
uv run --extra examples --extra torch-cu12 --with matplotlib \
    python -m playground.experiments.franka_allegro_grasp.visualize --num-envs 16 --random

# 특정 기능 비활성화
uv run --extra examples --extra torch-cu12 \
    python -m playground.experiments.franka_allegro_grasp.visualize \
    --no-depth --no-show-depth --no-debug-camera
```

### 로깅

```bash
# TensorBoard
tensorboard --logdir runs/

# Weights & Biases (기본 활성화, --no-wandb로 비활성화)
```

## Task 설명

**목표**: Franka arm + Allegro hand로 테이블 위의 큐브를 잡아서 목표 높이까지 들어올리기

### Task Phase (시각화용)

Phase는 시각화 및 로깅 목적으로 추적되지만, **보상은 DEXTRAH 원본처럼 continuous하게 계산**됩니다:

1. **Reach**: End effector가 큐브에 접근 (hand_to_object_dist < 0.3m)
2. **Grasp**: 손가락으로 큐브를 잡는 중 (cube_height > table + 0.05m)
3. **Lift**: 큐브를 목표 높이까지 들어올리기

## Scene Layout

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

| 항목 | 위치 | 설명 |
|------|------|------|
| Robot base | (0.0, 0.25, 0.0) | 테이블 Y축 중심에 맞춤 |
| Table center | (-0.3, -0.5) | 로봇 팔 도달 범위 내 |
| Table extends | X: [-0.6, 0.0], Y: [-0.9, -0.1] | 0.6m × 0.8m |
| Cube spawn | (-0.3, -0.5, 0.45) | 테이블 중심, 높이 0.45m |

### 다중 환경 (Vectorized)

환경은 Grid 형태로 복제됩니다:
- **Spacing**: (1.5m, 1.5m, 0.0m)
- **Grid 순서**: `grid_x = env_idx // grid_size`, `grid_y = env_idx % grid_size`

## Observation Space

DEXTRAH-G는 teacher와 student에 서로 다른 observation을 사용합니다.

### Student Observations (159D + Depth)

| 항목 | 차원 | 설명 |
|------|------|------|
| Robot DOF positions | 23 | 7 Franka + 16 Allegro |
| Robot DOF velocities | 23 | ×0.1 스케일링 |
| Hand keypoint positions | 15 | palm(3) + 4 fingertips(12) |
| Hand keypoint velocities | 15 | ×0.1 스케일링 |
| Goal position | 3 | 목표 위치 |
| Previous action | 11 | 이전 FABRICS action |
| Fabric state q | 23 | IK 출력 관절 위치 |
| Fabric state qd | 23 | IK 출력 관절 속도 (×0.1) |
| Fabric state qdd | 23 | IK 출력 관절 가속도 (×0.01) |
| **합계** | **159** | |
| Depth image | 1×120×160 | CNN 입력 (별도 텐서) |

### Teacher Observations (172D)

Teacher는 student observation + **privileged object information** (13D)을 사용합니다:

| 항목 | 차원 | 설명 |
|------|------|------|
| Student observations | 159 | 위와 동일 |
| Object position | 3 | 큐브 월드 좌표 |
| Object orientation | 4 | 큐브 쿼터니언 |
| Object linear velocity | 3 | 큐브 선속도 (×0.1) |
| Object angular velocity | 3 | 큐브 각속도 (×0.1) |
| **합계** | **172** | |

**핵심**: Student는 object 정보를 직접 받지 않습니다. 대신 depth image로부터 학습해야 합니다.

## Action Space (FABRICS, 11D)

DEXTRAH 원본과 동일한 FABRICS 기반 11D 액션 공간을 사용합니다:

| 인덱스 | 항목 | 차원 | 범위 (정규화 후) | 설명 |
|--------|------|------|-----------------|------|
| 0-2 | Palm XYZ | 3 | [-0.6, 0.0] × [-0.9, -0.1] × [0.35, 0.8] | EE 목표 위치 (m) |
| 3-5 | Palm RPY | 3 | [-π, π] × [-π/4, π/4] × [-π, π] | EE 목표 방향 (rad) |
| 6-10 | Hand PCA | 5 | [0.25, 3.83] × [-0.33, 3.00] × ... | PCA 좌표 |

### 변환 파이프라인

```
11D FABRICS Action ([-1, 1])
    ↓ normalize_action()
┌─────────────────────────────────────────────┐
│ 6D Palm Pose          │ 5D Hand PCA         │
│ (XYZ + RPY)           │                     │
└─────────────────────────────────────────────┘
    ↓                           ↓
Differential IK             PCA Projection
(Analytical Jacobian +      (5×16 DEXTRAH Matrix)
 Damped Least Squares)
    ↓                           ↓
7D Franka Joints        16D Allegro Joints
    ↓                           ↓
└──────────── 23D Joint Targets ─────────────┘
```

## Network Architecture

### Teacher (Asymmetric Actor-Critic)

```
Actor:  teacher_obs(172D) → LSTM(1024) → LayerNorm → MLP[512,512] → action(11D)
Critic: teacher_obs(172D) → LSTM(2048) → LayerNorm → MLP[1024,512] → value(1)
```

Actor와 Critic은 **완전히 분리된 네트워크**입니다 (asymmetric central value).

### Student (CNN + LSTM)

```
Depth(1×120×160) → CNN[16,32,64,128] → LayerNorm → AdaptiveAvgPool → Linear → 128D
                                                                        ↓
Concat(128D CNN, 159D proprio) → LSTM(512) → LayerNorm ──→ MLP[512,512,256] → action(11D)
                                                    └──→ AuxMLP[512,256]   → obj_pos(3D)
```

**Auxiliary Head**: LSTM 출력에서 object position을 예측하는 보조 task. Beta schedule (1.0 → 0.0)로 점진적으로 비활성화됩니다.

## Timing

DEXTRAH-G의 계층적 제어 주파수를 정확히 구현합니다:

```
Physics:  120Hz  (sim_dt = 1/120)
Fabric:    60Hz  (control_decimation = 2, 120/2)
Policy:    15Hz  (policy_decimation = 4, 60/4)
```

매 policy step에서 동일한 action이 4번의 fabric step에 적용됩니다.

## Reward Structure

DEXTRAH 원본의 continuous reward 구조를 사용합니다:

```
total_reward = hand_to_object + object_to_goal + finger_curl_reg + lift
```

| Component | 수식 | Weight | Sharpness |
|-----------|------|--------|-----------|
| `hand_to_object` | `w × exp(-s × max_dist(palm+fingertips, cube))` | 1.0 | 10.0 |
| `object_to_goal` | `w × exp(-s × dist_cube_to_goal)` | 5.0 | 15.0 |
| `lift` | `w × exp(-s × vertical_error)` | 5.0 | 8.5 |
| `finger_curl_reg` | `w × ‖q - curled_q‖²` | -0.01 | - |

## Distillation Loss

Student 학습 시 사용되는 loss:

```
L = L_action + β(t) · L_pos
```

| Component | 수식 | 설명 |
|-----------|------|------|
| `L_action` | `(1/σ²_teacher) · (μ_student - a_teacher)²` | Inverse-variance weighted action imitation |
| `L_pos` | `‖pos_pred - pos_gt‖²` | Auxiliary object position prediction |
| `β(t)` | `1.0 → 0.0` (15k iter) | Auxiliary loss weight schedule |

## 프로젝트 구조

```
franka_allegro_grasp/
├── __init__.py      # 패키지 export
├── config.py        # 환경/학습 설정
│                    #   EnvConfig, TeacherPPOConfig, StudentConfig, DistillConfig
├── networks.py      # 신경망 아키텍처
│                    #   TeacherActorCritic (LSTM actor-critic)
│                    #   StudentNetwork (CNN + LSTM + aux head)
│                    #   DepthEncoder (4-layer CNN + LayerNorm)
│                    #   LSTMBlock, RunningMeanStd
├── env.py           # FrankaAllegroGraspEnv
│                    #   159D student obs + 172D teacher obs
│                    #   Policy decimation (15Hz)
│                    #   Fabric state tracking (q, qd, qdd)
├── fabric.py        # FABRICS 구현
│                    #   FabricActionController: 11D → 23D 변환
│                    #   Analytical Geometric Jacobian (Franka DH parameters)
│                    #   GraspFabric: 파지 특징/보상 계산
│                    #   HAND_PCA_MATRIX: DEXTRAH 원본 5×16 PCA 행렬
├── train.py         # Stage 1: Teacher PPO (LSTM, sequence minibatch)
├── distill.py       # Stage 2: Student distillation
├── visualize.py     # Newton viewer 시각화 (teacher/student checkpoint 지원)
├── view_depth.py    # Depth 이미지 확인 스크립트
└── README.md
```

## 환경 설정

### Simulation

```python
fps: int = 120                # Physics at 120Hz
sim_substeps: int = 1
control_decimation: int = 2   # 60Hz fabric
policy_decimation: int = 4    # 15Hz policy (DEXTRAH-G)
episode_length: int = 600     # 10 seconds
```

### Depth Sensor (DEXTRAH-G)

```python
use_depth_sensor: bool = True
depth_width: int = 160
depth_height: int = 120
depth_fov: float = 48.0      # DEXTRAH-G: 48 degrees
depth_min: float = 0.5       # DEXTRAH-G: 0.5m
depth_max: float = 1.3       # DEXTRAH-G: 1.3m
```

**카메라 위치**: 테이블 짧은 면(로봇 쪽)에서 테이블/큐브를 바라보는 고정 뷰
- 위치: 각 환경의 (-0.3, 0.1, 0.43)
- 방향: 큐브 영역 (-0.3, -0.5, 0.45) 방향

### Robot Parameters

```python
# Franka Arm
franka_stiffness: float = 500.0
franka_damping: float = 100.0
franka_effort_limit: float = 80.0

# Allegro Hand
hand_stiffness: float = 200.0
hand_damping: float = 20.0
hand_effort_limit: float = 20.0
```

## Teacher PPO 하이퍼파라미터 (DEXTRAH-G)

| 항목 | DEXTRAH-G 원본 | 이 구현 |
|------|---------------|--------|
| **Actor** | LSTM(1024) + MLP[512,512] | LSTM(1024) + MLP[512,512] |
| **Critic** | LSTM(2048) + MLP[1024,512] | LSTM(2048) + MLP[1024,512] |
| learning_rate (actor) | 3e-4 | 3e-4 |
| learning_rate (critic) | 5e-5 | 5e-5 |
| lr_schedule | adaptive | adaptive |
| gamma | 0.998 | 0.998 |
| gae_lambda | 0.95 | 0.95 |
| clip_epsilon | 0.2 | 0.2 |
| kl_threshold | 0.016 | 0.016 |
| entropy_coef | 0.002 | 0.002 |
| value_coef | 4.0 | 4.0 |
| bounds_loss_coef | 0.005 | 0.005 |
| horizon_length | 16 | 16 |
| mini_epochs | 4 | 4 |
| minibatch_size | 16384 | 16384 |
| sequence_length | 16 | 16 |
| max_epochs | 20000 | 20000 |
| activation | elu | elu |
| normalize_value | True (RunningMeanStd) | True (RunningMeanStd) |

### Student Distillation 하이퍼파라미터

| 항목 | DEXTRAH-G 원본 | 이 구현 |
|------|---------------|--------|
| **CNN** | [16,32,64,128] + LayerNorm | [16,32,64,128] + LayerNorm |
| **LSTM** | 512 | 512 |
| **MLP** | [512,512,256] | [512,512,256] |
| **Aux head** | [512,256] → 3D | [512,256] → 3D |
| CNN output dim | 128 | 128 |
| learning_rate | 1e-4 | 1e-4 |
| warmup_steps | 1000 | 1000 |
| max_iterations | 100000 | 100000 |
| beta_initial | 1.0 | 1.0 |
| beta_final | 0.0 | 0.0 |
| beta_decay_iter | 15000 | 15000 |
| sequence_length | 20 | 20 |

## Analytical Jacobian (Franka DH Parameters)

IK에 사용되는 Jacobian은 Franka의 modified DH parameters를 이용한 **analytical geometric Jacobian**으로 계산됩니다:

```
Revolute joint i:
    J_v_i = z_i × (p_ee - p_i)   (linear velocity)
    J_w_i = z_i                    (angular velocity)
```

Franka modified DH parameters (Craig convention):

| Joint | a_{i-1} | d_i   | alpha_{i-1} |
|-------|---------|-------|-------------|
| 1     | 0       | 0.333 | 0           |
| 2     | 0       | 0     | -π/2        |
| 3     | 0       | 0.316 | π/2         |
| 4     | 0.0825  | 0     | π/2         |
| 5     | -0.0825 | 0.384 | -π/2        |
| 6     | 0       | 0     | π/2         |
| 7     | 0.088   | 0     | π/2         |
| flange| 0       | 0.107 | 0           |

## DEXTRAH-G 원본과의 주요 차이점

| 항목 | DEXTRAH-G 원본 | 이 구현 |
|------|---------------|--------|
| 로봇 | Kuka arm | Franka Emika Panda |
| 시뮬레이터 | Isaac Lab | Newton |
| IK 방식 | FABRICS geometric IK | Analytical Jacobian + Damped Least Squares |
| Palm Pose 범위 | Kuka workspace | Franka workspace에 맞게 조정 |
| Depth 정규화 | [0.5, 1.3]m → [0, 1] | 동일 |

## 참고 자료

- [DEXTRAH-G](https://github.com/NVlabs/DEXTRAH) - Kuka + Allegro grasping (원본 참조)
- [FABRICS](https://github.com/NVlabs/FABRICS) - Riemannian Geometric Fabrics
- [rl_games](https://github.com/Denys88/rl_games) - PPO LSTM 구현 참조
