# Franka Arm + Allegro Hand Combined Robot

Franka Emika Panda 로봇 팔에 Allegro 손을 결합한 manipulation 시스템입니다.

## 실행 방법

```bash
# 기본 실행 (GUI)
uv run --extra examples python -m playground.experiments.franka_allegro.example_franka_allegro

# Headless 모드
uv run --extra examples python -m playground.experiments.franka_allegro.example_franka_allegro --headless
```

## 로봇 구조

### Franka Emika Panda (FR3)

- **DOF**: 7 (revolute joints)
- **Base**: 원점에 고정
- **End Effector**: `fr3_link8`

### Allegro Hand (Left)

- **DOF**: 16 (4 fingers x 4 joints)
- **Attachment**: Franka EE에서 Z축으로 0.1m 오프셋, 180도 회전

## DOF 매핑

Newton은 URDF를 kinematic tree 순서로 로드합니다. Allegro hand의 DOF 매핑:

| DOF | 손가락 | URDF Links | 관절 구조 |
|-----|--------|------------|-----------|
| 0-3 | Ring (약지) | link_8-11 | Abduction, MCP, PIP, DIP |
| 4-7 | Middle (중지) | link_4-7 | Abduction, MCP, PIP, DIP |
| 8-11 | Index (검지) | link_0-3 | Abduction, MCP, PIP, DIP |
| 12-15 | Thumb (엄지) | link_12-15 | Pitch, Roll, MCP, PIP |

### 관절 타입

- **Abduction**: 손바닥에 가장 가까운 관절, 손가락을 좌우로 벌리는 동작
- **MCP** (Metacarpophalangeal): 손가락 굽힘의 첫 번째 관절
- **PIP** (Proximal Interphalangeal): 손가락 굽힘의 두 번째 관절
- **DIP** (Distal Interphalangeal): 손가락 끝 관절

### 관절 한계 (radians)

| 관절 타입 | Lower | Upper | Range |
|-----------|-------|-------|-------|
| Abduction | -0.470 | 0.470 | 0.940 |
| MCP | -0.196 | 1.610 | 1.806 |
| PIP | -0.174 | 1.709 | 1.883 |
| DIP | -0.227 | 1.618 | 1.845 |

## 관절 파라미터

### Franka Arm

```python
joint_target_ke = 500.0   # Stiffness
joint_target_kd = 100.0   # Damping
joint_effort_limit = 80.0
joint_armature = 0.1
```

### Allegro Hand

```python
joint_target_ke = 200.0   # Stiffness
joint_target_kd = 20.0    # Damping
joint_effort_limit = 20.0
joint_armature = 0.05
```

## Articulation 병합

두 URDF를 하나의 articulation으로 결합하는 핵심 코드:

```python
# 1. Allegro root joint의 parent를 Franka EE로 설정
allegro_root_joint_idx = joint_offset
builder.joint_parent[allegro_root_joint_idx] = ee_body_idx
builder.joint_X_p[allegro_root_joint_idx] = allegro_xform

# 2. 모든 Allegro joints를 같은 articulation에 할당
for j in range(joint_offset, builder.joint_count):
    builder.joint_articulation[j] = 0

# 3. Articulation 메타데이터 업데이트
builder.articulation_start = [0]
builder.articulation_key = ["franka_allegro"]
```

## Solver 설정

`SolverMuJoCo`를 사용하며 다관절 로봇에 적합한 파라미터를 설정합니다.

**기본 예제** (단일 환경):

```python
self.solver = newton.solvers.SolverMuJoCo(
    self.model,
    solver="newton",
    integrator="implicitfast",
    cone="elliptic",
    njmax=15000,
    nconmax=15000,
    iterations=15,
    ls_iterations=100,
    impratio=1000.0,
)
```

**Grasp 학습 환경** (다중 환경, 메모리 최적화):

```python
# njmax/nconmax을 환경 수에 비례하여 설정
# 큰 값(500*N)은 collision narrowphase에서 OOM 발생
self.solver = newton.solvers.SolverMuJoCo(
    self.model,
    solver="newton",
    integrator="implicitfast",
    njmax=30 * num_envs,    # ~30 contacts per env
    nconmax=20 * num_envs,  # ~20 constraints per env
    impratio=100.0,
    cone="elliptic",
    iterations=50,
    ls_iterations=100,
)
```

## CUDA Graph 비활성화

제어 타겟이 매 프레임 변경되므로 CUDA graph capture를 비활성화합니다:

```python
def capture(self):
    self.graph = None
```

---

# DEXTRAH-G Grasping (franka_allegro_grasp)

[DEXTRAH-G](https://github.com/NVlabs/DEXTRAH)의 2단계 teacher-student distillation 파이프라인 구현입니다.
코드: `playground/experiments/franka_allegro_grasp/`

## 알고리즘 개요

```
Stage 1: Teacher (Privileged RL)          Stage 2: Student Distillation
+------------------------------+          +---------------------------------+
| Full state (172D)            |          | Depth image (120x160)           |
| +                            |          | + Proprioception (159D)         |
| ObsRunningMeanStd normalize  |          | +                               |
| +                            |          | +                               |
| LSTM(1024) -> [concat obs]   |  --->    | CNN -> LSTM(512) -> MLP         |
|   -> MLP[512,512] -> act(11) | freeze   | +                               |
| MLP[1024,512] -> [concat obs]| teacher  | L_action + beta * L_pos         |
|   -> LSTM(2048) -> value(1)  |          | Imitate teacher + predict obj   |
+------------------------------+          +---------------------------------+
```

## 실행 방법

### Teacher 학습

```bash
# 기본 (256 envs)
uv run --extra examples --extra torch-cu12 python -m playground.experiments.franka_allegro_grasp.train \
    --num-envs 256 --no-wandb

# 빠른 테스트
uv run --extra examples --extra torch-cu12 python -m playground.experiments.franka_allegro_grasp.train \
    --num-envs 64 --max-iterations 100 --no-wandb

# Checkpoint resume
uv run --extra examples --extra torch-cu12 python -m playground.experiments.franka_allegro_grasp.train \
    --checkpoint checkpoints/teacher_*/best.pt
```

### Student Distillation

```bash
uv run --extra examples --extra torch-cu12 python -m playground.experiments.franka_allegro_grasp.distill \
    --teacher-checkpoint checkpoints/teacher_*/best.pt
```

### 시각화

```bash
# 기본 실행
uv run --extra examples --extra torch-cu12 --with matplotlib \
    python -m playground.experiments.franka_allegro_grasp.visualize

# 학습된 정책
uv run --extra examples --extra torch-cu12 --with matplotlib \
    python -m playground.experiments.franka_allegro_grasp.visualize \
    --checkpoint checkpoints/teacher_*/best.pt
```

## Network Architecture

### Teacher Actor-Critic (rl_games 스타일)

```
Actor (before_mlp=True, concat_output=True):
  obs(172D) -> LSTM(1024) -> LayerNorm -> [concat obs(172D)]
    -> MLP[512,512](ELU) -> action_mean(11D)
  MLP input: 1024 + 172 = 1196D

Critic (before_mlp=False, concat_input=True, concat_output=True):
  obs(172D) -> MLP[1024,512](ELU) -> [concat obs(172D)]
    -> LSTM(2048) -> LayerNorm -> [concat obs(172D)] -> value(1)
  LSTM input: 512 + 172 = 684D
  Value head input: 2048 + 172 = 2220D
```

- Actor와 Critic은 완전히 분리된 네트워크 (asymmetric central value)
- **concat_output** (rl_games): LSTM/MLP 출력에 원본 obs를 concat하는 skip connection
- **concat_input** (critic): MLP 출력에 원본 obs를 concat하여 LSTM 입력 구성
- Actor: LSTM → concat obs → MLP → action
- Critic: MLP → concat obs → LSTM → concat obs → value
- Total: Actor 5.8M + Critic 23.1M = **28.8M params**

### Student (CNN + LSTM)

```
Depth(1x120x160) -> CNN[16,32,64,128] -> LayerNorm -> AdaptiveAvgPool -> Linear -> 128D
                                                                          |
Concat(128D CNN, 159D proprio) -> LSTM(512) -> LayerNorm -> MLP[512,512,256] -> action(11D)
                                                     +----> AuxMLP[512,256]  -> obj_pos(3D)
```

## Observation Space

### Student (159D + Depth)

| 항목 | 차원 | 비고 |
|------|------|------|
| Robot DOF positions | 23 | 7 Franka + 16 Allegro |
| Robot DOF velocities | 23 | x0.1 스케일링 |
| Hand keypoint positions | 15 | palm(3) + 4 fingertips(12) |
| Hand keypoint velocities | 15 | x0.1 스케일링 |
| Goal position | 3 | |
| Previous action | 11 | FABRICS 11D |
| Fabric state q | 23 | IK 출력 관절 위치 |
| Fabric state qd | 23 | x0.1 스케일링 |
| Fabric state qdd | 23 | x0.01 스케일링 |
| **합계** | **159** | |
| Depth image | 1x120x160 | CNN 입력 (별도 텐서) |

### Teacher (172D = Student 159D + Privileged 13D)

| 추가 항목 | 차원 |
|-----------|------|
| Object position | 3 |
| Object orientation (quat) | 4 |
| Object linear velocity | 3 (x0.1) |
| Object angular velocity | 3 (x0.1) |

### Observation Normalization

`ObsRunningMeanStd`로 per-feature running statistics 정규화 후 [-5, 5] 클리핑:
```python
normalized_obs = (raw_obs - running_mean) / sqrt(running_var + eps)
normalized_obs = clamp(normalized_obs, -5.0, 5.0)
```

## Action Space (FABRICS 11D)

| 인덱스 | 항목 | 범위 (정규화 후) |
|--------|------|-----------------|
| 0-2 | Palm XYZ | [-0.6, 0.0] x [-0.9, -0.1] x [0.35, 0.8] (m) |
| 3-5 | Palm RPY | [-pi, pi] x [-pi/4, pi/4] x [-pi, pi] (rad) |
| 6-10 | Hand PCA | 5D PCA -> 16D joints (DEXTRAH PCA matrix) |

```
11D FABRICS Action ([-1, 1])
    | normalize_action()
    v
6D Palm Pose (XYZ+RPY)       5D Hand PCA
    |                              |
    v                              v
Differential IK              PCA Projection
(Analytical Jacobian +       (5x16 DEXTRAH Matrix)
 Damped Least Squares)
    |                              |
    v                              v
7D Franka Joints          16D Allegro Joints
          |
          v
     23D Joint Targets
```

## Reward Structure

```
total = (hand_to_object + object_to_goal + lift + finger_curl_reg) * 0.01
```

| Component | 수식 | Weight | Sharpness |
|-----------|------|--------|-----------|
| `hand_to_object` | `w * exp(-s * max_dist(hand_points, cube))` | 1.0 | 10.0 |
| `object_to_goal` | `w * exp(-s * dist_cube_to_goal)` | 5.0 | 15.0 (ADR: -15→-20) |
| `lift` | `w * exp(-s * vertical_error)` | 5.0 (ADR: 5→0) | 8.5 |
| `finger_curl_reg` | `w * norm(q - curled_q)^2` | -0.01 (ADR: -0.01→-0.005) | - |
| **reward_scale** | 모든 reward에 곱함 | **0.01** | - |

참고: `in_success_region_at_rest_weight=10.0`은 config에 정의되어 있지만 실제 reward 계산에는 사용되지 않음 (ADR/distillation 추적용).

## Timing

```
Physics:  120Hz  (sim_dt = 1/120)
Fabric:    60Hz  (control_decimation = 2)
Policy:    15Hz  (policy_decimation = 4)
Episode:   10s   (600 steps at 60Hz)
```

## Teacher PPO 하이퍼파라미터

원본 DEXTRAH-G와의 비교 (현재 config.py 기준):

| 항목 | DEXTRAH-G 원본 | 이 구현 | 비고 |
|------|---------------|--------|------|
| num_envs | 4096 | 256 | VRAM 제한 |
| **Actor** | LSTM(1024) -> [concat obs] -> MLP | 동일 | skip connection |
| **Critic** | MLP -> [concat obs] -> LSTM(2048) | 동일 | MLP-first 순서 |
| learning_rate (actor) | 3e-4 | 3e-4 | |
| learning_rate (critic) | 5e-5 | 5e-5 | |
| lr_schedule | linear | linear | LR → 0 over training |
| gamma | 0.998 | 0.998 | |
| kl_threshold | 0.013 | 0.013 | |
| entropy_coef | 0.002 | 0.002 | |
| value_coef | 4.0 | 4.0 | |
| bounds_loss_coef | 0.005 | 0.005 | |
| horizon_length | 16 | 16 | |
| mini_epochs | 4 | 4 | |
| minibatch_size | 16384 | 1024 | 4 minibatch/epoch 유지 |
| sequence_length | 16 | 16 | |
| normalize_input | True | True | ObsRunningMeanStd |
| normalize_value | True | True | scalar RunningMeanStd |
| reward_scale | 0.01 | 0.01 | |
| value_bootstrap | False | True | 개선: truncation 처리 |

## 원본 DEXTRAH-G와의 차이점

### 일치하는 부분 (원본 소스 `/home/ai5090/Documents/DEXTRAH/` 기반 검증)
- Actor: LSTM → concat_output → MLP (before_mlp=True, concat_output=True)
- Critic: MLP → concat_input → LSTM → concat_output → value (before_mlp=False)
- Reward 구조: 4 components (hand_to_object, object_to_goal, finger_curl_reg, lift) x 0.01 scaling
- Observation/Value normalization (RunningMeanStd)
- LR schedule: linear (0으로 선형 감소)
- Episode termination: XY workspace boundary + Z fall + timeout (terminated/truncated 분리)
- PCA hand mapping (5D -> 16D, 원본 matrix 사용)
- Timing hierarchy (120/60/15Hz)
- Minibatch당 epoch 수 (4)

### 남아있는 차이
| 항목 | 원본 | 현재 | 영향도 |
|------|------|------|--------|
| num_envs | 4096 | 256 | 높음 (VRAM 제한) |
| value_bootstrap | False | True | 개선 (truncation GAE) |
| Observation noise | per-step + bias noise | 없음 | 중간 |
| Asymmetric critic | clean obs for critic | 동일 obs | 중간 |
| ADR | 50-stage curriculum | 없음 | 중간 (sim-to-real) |
| IK | Full FABRICS geometric | Analytical Jacobian + DLS | 중간 |
| Physics engine | IsaacGym (PhysX) | Newton (MuJoCo) | 기본 차이 |
| Object diversity | 140 objects + one-hot | 단일 큐브 | 낮음 (단일 task) |

## Analytical Jacobian (Franka DH Parameters)

Franka modified DH parameters (Craig convention):

| Joint | a_{i-1} | d_i   | alpha_{i-1} |
|-------|---------|-------|-------------|
| 1     | 0       | 0.333 | 0           |
| 2     | 0       | 0     | -pi/2       |
| 3     | 0       | 0.316 | pi/2        |
| 4     | 0.0825  | 0     | pi/2        |
| 5     | -0.0825 | 0.384 | -pi/2       |
| 6     | 0       | 0     | pi/2        |
| 7     | 0.088   | 0     | pi/2        |
| flange| 0       | 0.107 | 0           |

```
Revolute joint i:
    J_v_i = z_i x (p_ee - p_i)   (linear velocity)
    J_w_i = z_i                    (angular velocity)
```

## 프로젝트 구조

```
franka_allegro/                     # 기본 로봇 예제
  example_franka_allegro.py
  README.md                         # 이 파일

franka_allegro_grasp/               # DEXTRAH-G 학습
  __init__.py
  config.py          # EnvConfig, TeacherPPOConfig, StudentConfig, DistillConfig
  networks.py        # TeacherActorCritic, StudentNetwork, ObsRunningMeanStd
  env.py             # FrankaAllegroGraspEnv (159D/172D obs, reward x0.01)
  fabric.py          # FabricActionController (11D->23D), Analytical Jacobian
  train.py           # Stage 1: Teacher LSTM PPO (obs normalization, skip connection)
  distill.py         # Stage 2: Student distillation
  visualize.py       # Newton viewer (teacher/student checkpoint)
  view_depth.py      # Depth 이미지 디버그
```

## 참고 자료

- [DEXTRAH-G](https://github.com/NVlabs/DEXTRAH) - Kuka + Allegro grasping (원본)
- [FABRICS](https://github.com/NVlabs/FABRICS) - Riemannian Geometric Fabrics
- [rl_games](https://github.com/Denys88/rl_games) - PPO LSTM 구현 참조
- [Franka Emika Panda](https://www.franka.de/)
- [Wonik Allegro Hand](https://www.wonikrobotics.com/robot-hand-allegro-hand)
