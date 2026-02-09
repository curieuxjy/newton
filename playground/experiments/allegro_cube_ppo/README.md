# Allegro Hand Cube Rotation with PPO

Newton 물리 엔진을 사용한 Allegro Hand 큐브 회전 강화학습 예제입니다.
IsaacGymEnvs의 [DextrEme](https://github.com/isaac-sim/IsaacGymEnvs/tree/main/isaacgymenvs/tasks/dextreme)를 참고하여 구현되었습니다.

## 프로젝트 구조

```
allegro_cube_ppo/
├── __init__.py      # 패키지 export
├── config.py        # 하이퍼파라미터 설정 (DextrEme 스타일)
├── env.py           # AllegroHandCubeEnv (Newton 기반 환경)
├── ppo.py           # PPO 알고리즘 (CleanRL 스타일)
├── train.py         # 학습 스크립트
├── visualize.py     # Newton viewer로 시각화
└── README.md
```

## Task 설명

**목표**: Allegro Hand로 큐브를 잡고 목표 방향(quaternion)으로 회전시키기

### Observation (49 dims)

| 항목 | 차원 | 설명 |
|------|------|------|
| Joint positions (normalized) | 16 | 손가락 관절 위치 [-1, 1] |
| Joint velocities (×0.2) | 16 | 손가락 관절 속도 |
| Cube relative position | 3 | 손바닥 기준 큐브 위치 |
| Cube orientation | 4 | 큐브 쿼터니언 (x,y,z,w) |
| Cube linear velocity (×0.2) | 3 | 큐브 선속도 |
| Cube angular velocity (×0.2) | 3 | 큐브 각속도 |
| Goal orientation | 4 | 목표 쿼터니언 |

### Action (16 dims)

- **Delta position control**: 현재 위치에서의 변화량
- Policy 출력은 [-1, 1]로 클램핑
- 최종 범위: [-1, 1] × action_scale (0.5)
- joint limits로 추가 클램핑

### Reward (DextrEme 스타일)

```python
reward = rot_reward + dist_penalty - action_penalty - action_delta_penalty - vel_penalty + success_bonus + fall_penalty
```

| 항목 | 공식 | 설명 |
|------|------|------|
| **Rotation reward** | `1.0 / (rot_dist + 0.1)` | 쿼터니언 정렬 보상 |
| **Distance penalty** | `-10.0 × dist` | 큐브가 손에서 멀어지면 페널티 |
| **Action penalty** | `-0.0002 × Σ(a²)` | 액션 크기 페널티 |
| **Action delta penalty** | `-0.0001 × Σ(Δa²)` | 액션 변화율 페널티 |
| **Velocity penalty** | `-0.05 × Σ((v/4)²)` | 관절 속도 페널티 |
| **Success bonus** | `+250` | N회 연속 성공 시 |
| **Fall penalty** | `-50` | 큐브 낙하/이탈 시 |

**Rotation distance (DextrEme 방식)**:

```python
q_diff = q_cube × conjugate(q_goal)
rot_dist = 2 × arcsin(||q_diff.xyz||)  # radians
```

### Success Condition

- Rotation distance < `success_tolerance` (0.4 rad ≈ 23°)
- `consecutive_successes` (5) 회 연속 유지
- 성공 시 새로운 랜덤 목표 생성

### Termination

- Episode timeout (400 steps = 8초)
- 큐브 낙하 (z < 0.05m)
- 큐브 이탈 (dist > 0.3m)

## 실행 방법

### 1. 시각화 (Newton Viewer)

```bash
# 랜덤 액션으로 시뮬레이터 확인
uv run --extra examples --extra torch-cu12 python -m playground.experiments.allegro_cube_ppo.visualize --num-envs 4

# Collision shapes 표시 (기본: 숨김)
uv run --extra examples --extra torch-cu12 python -m playground.experiments.allegro_cube_ppo.visualize --show-collision

# 학습된 정책으로 시각화
uv run --extra examples --extra torch-cu12 python -m playground.experiments.allegro_cube_ppo.visualize \
    --checkpoint checkpoints/allegro_cube_ppo_XXXXXX/final.pt
```

**Orientation Frame Visualization**:

- 큐브의 현재 orientation을 3축 좌표계(RGB = XYZ)로 표시합니다
- 목표 orientation도 동일한 방식으로 각 환경 위에 표시됩니다
- 실제 큐브가 goal orientation에 5회 연속 도달하면 새로운 랜덤 목표가 생성됩니다
- 축 색상: **빨강(X)**, **초록(Y)**, **파랑(Z)**

## Multi-Environment 시각화 주의사항

Newton에서 다중 환경을 시각화할 때 좌표계 처리에 주의가 필요합니다.

### World Offset 메커니즘

Newton의 viewer는 두 가지 방식으로 다중 환경을 배치합니다:

1. **Physical Spacing** (`builder.replicate()`의 `spacing` 파라미터)
   - 시뮬레이션 내 실제 물리적 위치 오프셋
   - `body_q` 등 상태 배열에 이미 반영됨
2. **Visual World Offset** (`viewer.set_world_offsets()`)
   - 렌더링 시에만 적용되는 시각적 오프셋
   - `set_model()` 호출 시 자동으로 계산됨

### 문제점

`viewer.log_lines()` 등 커스텀 시각화 함수는 **world offset을 자동 적용하지 않습니다**.
따라서 `replicate()`로 physical spacing을 설정한 후, viewer가 추가로 visual offset을 적용하면
커스텀 시각화 좌표가 렌더링된 모델 위치와 불일치합니다.

### 해결책

Physical spacing을 사용하는 경우, viewer의 world offset을 비활성화합니다:

```python
# 방법 1: Physical spacing 사용 시 visual offset 비활성화 (권장)
builder.replicate(hand_builder, num_envs, spacing=(0.5, 0.5, 0.0))
model = builder.finalize()
viewer.set_model(model)
viewer.set_world_offsets((0.0, 0.0, 0.0))  # visual offset 비활성화

# 이제 body_q 좌표를 그대로 log_lines()에 사용 가능
body_pos = state.body_q.numpy()[body_idx, :3]
viewer.log_lines("/my_lines", starts, ends, colors)
```

```python
# 방법 2: Visual offset만 사용 (physical spacing 없이)
builder.replicate(hand_builder, num_envs, spacing=(0.0, 0.0, 0.0))
model = builder.finalize()
viewer.set_model(model)  # 자동으로 visual offset 계산

# log_lines() 호출 시 world_offset을 수동으로 더해야 함
world_offset = viewer.world_offsets.numpy()[env_idx]
adjusted_pos = body_pos + world_offset
viewer.log_lines("/my_lines", adjusted_starts, adjusted_ends, colors)
```

### 내장 시각화 vs 커스텀 시각화

| 함수 | World Offset 자동 적용 |
|------|----------------------|
| `viewer.log_state()` | O (shapes, bodies) |
| `viewer.log_contacts()` | O |
| `viewer.log_joint_basis()` | O |
| `viewer.log_lines()` | **X** |
| `viewer.log_shapes()` | **X** |

커스텀 시각화 시에는 위 테이블을 참고하여 좌표를 적절히 조정해야 합니다.

### 2. 학습

```bash
# 기본 학습 (4096 envs)
uv run --extra examples --extra torch-cu12 python -m playground.experiments.allegro_cube_ppo.train \
    --num-envs 4096 \
    --total-timesteps 100000000 \

# 빠른 테스트
uv run --extra examples --extra torch-cu12 python -m playground.experiments.allegro_cube_ppo.train \
    --num-envs 256 \
    --total-timesteps 1000000 \
    --no-wandb
```

### 3. 로깅

```bash
# TensorBoard (기본 활성화)
tensorboard --logdir runs/

# Wandb
pip install wandb && wandb login
uv run --extra examples --extra torch-cu12 python -m playground.experiments.allegro_cube_ppo.train \
    --wandb-project newton-allegro-cube
```

## 하이퍼파라미터

### EnvConfig

```python
# Simulation
fps: int = 60
sim_substeps: int = 2
control_decimation: int = 2      # 30Hz control
episode_length: int = 400        # 8 seconds

# Robot
hand_stiffness: float = 40.0
hand_damping: float = 2.0
action_scale: float = 0.5

# Reward (DextrEme style)
rot_reward_scale: float = 1.0
rot_eps: float = 0.1
dist_reward_scale: float = -10.0
action_penalty_scale: float = 0.0002
velocity_penalty_scale: float = 0.05

# Success/Failure
success_tolerance: float = 0.4   # radians (~23°)
consecutive_successes: int = 5
reach_goal_bonus: float = 250.0
fall_penalty: float = -50.0
```

### PPOConfig

```python
learning_rate: float = 3e-4
gamma: float = 0.99
gae_lambda: float = 0.95
clip_epsilon: float = 0.2
entropy_coef: float = 0.01
value_coef: float = 0.5
bounds_loss_coef: float = 0.0001    # 액션 범위 초과 페널티
num_epochs: int = 5
rollout_steps: int = 24
hidden_dims: tuple = (512, 256, 128)
```

### Policy Network 안정화

```python
# Actor log_std 초기화 및 범위 제한
actor_log_std = nn.Parameter(torch.full((num_actions,), -0.5))  # σ ≈ 0.6
log_std_clamped = torch.clamp(actor_log_std, min=-2.0, max=0.5)  # σ ∈ [0.14, 1.65]

# Value loss clipping (PPO2 스타일)
value_clipped = old_values + clamp(new_values - old_values, -ε, +ε)
value_loss = max(unclipped_loss, clipped_loss)

# Bounds loss - 액션 범위 초과 페널티
bounds_loss = mean(clamp(|actions| - 1, 0)²)
```

- **Entropy 범위**: 약 10~30 (16 actions 기준)
- **Reward clipping**: [-100, 100] 범위로 제한

## 참고 자료

- [IsaacGymEnvs DextrEme](https://github.com/isaac-sim/IsaacGymEnvs/tree/main/isaacgymenvs/tasks/dextreme)
- [IsaacLab Allegro Hand](https://github.com/isaac-sim/IsaacLab/blob/main/source/isaaclab_tasks/isaaclab_tasks/direct/allegro_hand/)
- [Newton Examples](https://github.com/nvidia-warp/newton)
