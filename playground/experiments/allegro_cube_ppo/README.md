# Allegro Hand Cube Rotation with PPO

Newton 물리 엔진을 사용한 Allegro Hand 큐브 회전 강화학습 예제입니다.
IsaacGymEnvs의 [DextrEme ManualDR](https://github.com/isaac-sim/IsaacGymEnvs/tree/main/isaacgymenvs/tasks/dextreme)를 참고하여 구현되었습니다.

## 프로젝트 구조

```
allegro_cube_ppo/
├── __init__.py      # 패키지 export
├── adr.py           # Vectorised ADR (DextrEme paper Section 4)
├── config.py        # 하이퍼파라미터 설정 (DextrEme ManualDR 정렬)
├── env.py           # AllegroHandCubeEnv (Newton 기반 환경)
├── ppo.py           # PPO 알고리즘 (LSTM + 분리 actor/critic)
├── rna.py           # Random Network Adversary (DextrEme Section B.3)
├── train.py         # 학습 스크립트
├── visualize.py     # Newton viewer로 시각화
└── README.md
```

## Task 설명

**목표**: Allegro Hand로 큐브를 잡고 목표 방향(quaternion)으로 회전시키기

### Observation (72 dims)

| 항목 | 차원 | 설명 |
|------|------|------|
| Joint positions (normalized) | 16 | 손가락 관절 위치 [-1, 1] (unscale) |
| Joint velocities | 16 | 손가락 관절 속도 (MLP 보상용, LSTM 대체) |
| Object pose (relative) | 7 | 손 기준 큐브 위치(3) + 쿼터니언(4) |
| Object velocities | 6 | 큐브 선속도(3) + 각속도×0.2(3) (MLP 보상용) |
| Goal pose (relative) | 7 | 손 기준 목표 위치(3) + 목표 쿼터니언(4) |
| Goal relative rotation | 4 | quat_mul(obj_rot, conjugate(goal_rot)) |
| Last actions | 16 | 이전 스텝의 액션 |

DextrEme ManualDR 원본 actor inputs (50 dims)에 velocity 정보(22 dims)를 추가하여
LSTM 없이 MLP만으로 학습할 수 있도록 보상합니다.

**DR 적용 시 관측 처리:**
- `dof_pos`: joint space에서 affine noise 적용 후 unscale (DextrEme 원본 순서)
- `object_pose`: affine noise → random pose injection → refresh rate gate → exponential delay
- `goal_relative_rot`: delayed `cube_rot_obs` 사용 (actor는 noisy observation을 봄)
- Reward 계산은 raw (non-delayed) 상태를 사용

### Action (16 dims)

- **Absolute position control + EMA smoothing** (DextrEme ManualDR 스타일)
- Policy 출력 [-1, 1] → `scale()` → [joint_lower, joint_upper]
- EMA: `cur_targets = ema × targets + (1 - ema) × prev_targets`
- EMA schedule: 0.2 → 0.15 over 1M frames (DextrEme paper Section 4)

### Reward (DextrEme compute_hand_reward 정확 재현)

```python
reward = dist_rew + rot_rew + action_penalty + action_delta_penalty
       + velocity_penalty + reach_goal_rew + fall_rew + timeout_rew
```

| 항목 | 공식 | Scale |
|------|------|-------|
| **Distance reward** | `dist × scale` | -10.0 |
| **Rotation reward** | `1.0 / (rot_dist + eps) × scale` | 1.0 (eps=0.1) |
| **Action penalty** | `scale × Σ(a²)` | -0.0001 |
| **Action delta penalty** | `scale × Σ((cur_targets - prev_targets)²)` | -0.01 |
| **Velocity penalty** | `-0.05 × Σ((v / (max_vel - tol))²)` | hardcoded |
| **Reach goal bonus** | goal 달성 시 | +250 |
| **Fall penalty** | dist ≥ fall_dist 시 | 0.0 (disabled) |
| **Timeout penalty** | timeout 시 `0.5 × fall_penalty` | 0.0 |

**Rotation distance (DextrEme 방식)**:
```python
q_diff = q_cube × conjugate(q_goal)
rot_dist = 2 × arcsin(||q_diff.xyz||)  # radians
```

### Success Condition (DextrEme hold_count 방식)

- Rotation distance < `success_tolerance` (0.4 rad ≈ 23°) → hold_count 증가
- hold_count > `num_success_hold_steps` (1) → 1 goal 달성, 새 목표 생성
- `successes` ≥ 50 → 에피소드 리셋
- 목표 근접 시 `progress_buf = 0` (에피소드 연장, DextrEme 원본)

### Termination

- Episode timeout (240 steps = 8초 at 30Hz control)
- 큐브 이탈 (dist > 0.24m from goal position)
- 50개 goal 모두 달성

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
- 실제 큐브가 goal orientation에 도달하면 새로운 랜덤 목표가 생성됩니다
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
# 기본 학습 (4096 envs, LSTM, ADR enabled)
uv run --extra examples --extra torch-cu12 python -m playground.experiments.allegro_cube_ppo.train \
    --num-envs 4096 \
    --total-timesteps 100000000

# MLP network (LSTM 대신)
uv run --extra examples --extra torch-cu12 python -m playground.experiments.allegro_cube_ppo.train \
    --num-envs 4096 \
    --network-type mlp

# DR 모드 선택
uv run --extra examples --extra torch-cu12 python -m playground.experiments.allegro_cube_ppo.train \
    --dr-mode adr      # Vectorised ADR (기본값)
    # --dr-mode static # 고정 범위 DR
    # --dr-mode off    # DR 없음

# 빠른 테스트
uv run --extra examples --extra torch-cu12 python -m playground.experiments.allegro_cube_ppo.train \
    --num-envs 256 \
    --total-timesteps 1000000 \
    --dr-mode off \
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

**로그 구조:**

| 카테고리 | 항목 | 설명 |
|----------|------|------|
| `episode/` | `reward_mean` | 에피소드 평균 보상 |
| | `length_mean` | 에피소드 평균 길이 |
| | `successes` | 전체 환경 성공 합계 (정수) |
| | `consecutive_successes` | EMA 연속 성공 (정수) |
| | `adr_rollout_perf` | ADR rollout worker 성능 EMA |
| | `adr_boundary_changes` | ADR boundary 변경 누적 횟수 |
| `train/` | `policy_loss`, `value_loss`, `entropy`, `kl_divergence` | PPO 학습 메트릭 |
| `reward/` | `rot_rew`, `dist_rew`, `action_penalty`, ... | 보상 구성 요소 |
| `adr/lower/` | `hand_stiffness`, `cube_mass`, `action_latency`, ... | 각 ADR 파라미터 하한 |
| `adr/upper/` | `hand_stiffness`, `cube_mass`, `action_latency`, ... | 각 ADR 파라미터 상한 |
| `perf/` | `fps` | 학습 처리 속도 |

## DextrEme 원본 코드 정렬

이 구현은 [IsaacGymEnvs DextrEme](https://github.com/isaac-sim/IsaacGymEnvs)의 원본 코드와
1:1 비교 검증되었습니다.

### Simulation

| 항목 | DextrEme 원본 | 이 구현 |
|------|-------------|--------|
| Physics frequency | 120 Hz (dt=1/60, substeps=2) | 120 Hz (fps=60, substeps=2) |
| Control frequency | 30 Hz (controlFreqInv=2) | 30 Hz (decimation=2) |
| Episode length | 240 steps (8s at 30Hz) | 240 steps |
| Max consecutive successes | 50 goals | 50 goals |
| Action control | Absolute + EMA | Absolute + EMA |
| EMA schedule | 0.2 → 0.15 over 1M frames | 0.2 → 0.15 over 1M frames |

### PPO Hyperparameters

| 항목 | DextrEme 원본 | 이 구현 |
|------|-------------|--------|
| actor_learning_rate | 1e-4 (linear schedule) | 1e-4 (linear schedule) |
| critic_learning_rate | 5e-5 (fixed) | 5e-5 (fixed) |
| gamma | 0.998 | 0.998 |
| tau (gae_lambda) | 0.95 | 0.95 |
| e_clip | 0.2 | 0.2 |
| entropy_coef | 0.0 | 0.0 |
| critic_coef (value_coef) | 4.0 | 4.0 |
| bounds_loss_coef | 0.0001 | 0.0001 |
| horizon_length (BPTT) | 16 | 16 |
| mini_epochs | 4 | 4 |
| minibatch_size | 16384 | 16384 |
| normalize_value | True | True (RunningMeanStd) |
| normalize_input | True | True |
| normalize_advantage | True | True |

### PPO Algorithm (rl_games)

| 항목 | rl_games 원본 | 이 구현 |
|------|-------------|--------|
| KL divergence | Analytical Gaussian | Analytical Gaussian |
| Value loss | Clipped | Clipped |
| Bounds loss target | Action mean (μ) | Action mean (μ) |
| Bounds loss soft bound | 1.1 | 1.1 |
| Value normalization | RunningMeanStd | RunningMeanStd |
| log_std clamp | [-5.0, 2.0] | [-5.0, 2.0] |
| Actor/Critic optimizers | Separate | Separate |

### Reward Structure

| 항목 | DextrEme 원본 | 이 구현 |
|------|-------------|--------|
| rot_reward_scale | 1.0 | 1.0 |
| rot_eps | 0.1 | 0.1 |
| dist_reward_scale | -10.0 | -10.0 |
| action_penalty_scale | -0.0001 | -0.0001 |
| action_delta_penalty_scale | -0.01 | -0.01 |
| velocity_penalty (hardcoded) | -0.05, denom=4.0 | -0.05, denom=4.0 |
| reach_goal_bonus | 250.0 | 250.0 |
| fall_penalty | 0.0 (disabled) | 0.0 |
| fall_dist | 0.24 | 0.24 |
| success_tolerance | 0.4 rad | 0.4 rad |
| num_success_hold_steps | 1 | 1 |

### Network Architecture (DextrEme paper Section 4)

| 항목 | DextrEme 원본 | 이 구현 |
|------|-------------|--------|
| Actor | LSTM(1024) + LayerNorm + MLP [512, 512] | LSTM(1024) + LayerNorm + MLP [512, 512] |
| Critic | LSTM(2048) + LayerNorm + MLP [1024, 512] | LSTM(2048) + LayerNorm + MLP [1024, 512] |
| Activation | ELU | ELU |
| LayerNorm | Yes | Yes |
| Actor/Critic 분리 | Separate networks + separate LR | Separate networks + separate LR |
| Observations | 50 dims (no velocity) | 72 dims (+ velocity) |
| init_sigma | 0.0 (σ=1.0) | 0.0 (σ=1.0) |
| MLP fallback | N/A | MLP [512, 512, 256, 128] (`--network-type mlp`) |

### Vectorised ADR (DextrEme paper Section 4)

| 항목 | DextrEme 원본 | 이 구현 |
|------|-------------|--------|
| Boundary fraction | 40% | 40% |
| Queue length (N) | 256 | 256 |
| Threshold high (tH) | 20 | 20 |
| Threshold low (tL) | 5 | 5 |
| Clear other queues | Yes | Yes |
| DR parameters | 22 (physics 7 + non-physics 15) | 22 (동일) |
| Objective metric | consecutive successes | consecutive successes |
| Default mode | ADR | ADR (`--dr-mode adr`) |

### Non-Physics Randomisations (DextrEme paper Section B)

**Action delay pipeline** (DextrEme `apply_actions` + `apply_action_noise_latency` 순서):
```
raw_action → FIFO queue → latency select → Bernoulli freeze(초기 dof_pos) → affine noise → RNA → clamp
```

**Observation delay pipeline** (DextrEme `compute_observations` 순서):
```
raw_cube_pose → affine noise → random pose injection → refresh rate gate → Bernoulli freeze
raw_dof_pos → affine noise (joint space) → unscale
```

| 항목 | DextrEme 원본 | 이 구현 |
|------|-------------|--------|
| **Stochastic Delays** | | |
| Cube obs exponential delay | persistent `obs_object_pose` buffer | persistent `obs_cube_pose` buffer |
| Action exponential delay | frozen value = initial dof_pos (per episode) | frozen value = initial dof_pos (per episode) |
| Action latency FIFO | `prev_actions_queue[:, 1:] = queue[:, :-1]` | `_action_queue[:, 1:] = queue[:, :-1]` |
| Discrete ADR sampling | `round(adr_value + (-(rand - 0.5)))` | `round(adr_value + (-(rand - 0.5)))` |
| Refresh rate gate | `(frame + offset) % rate == 0` | `(progress_buf + offset) % rate == 0` |
| Refresh offset | `round(rand * rate - 0.5)` per-env | `round(rand * rate - 0.5)` per-env |
| **Correlated & Uncorrelated Noise** | | |
| Affine action noise | `scaling × a + additive + white` | `scaling × a + additive + white` |
| Affine cube pose noise | `scaling × p + additive + white` | `scaling × p + additive + white` |
| Affine dof_pos noise | `unscale(scaling × q + additive + white)` | `unscale(scaling × q + additive + white)` |
| Gaussian stdev mapping | `exp(value²) - 1` | `exp(value²) - 1` |
| Correlated noise resampling | Per episode (on reset) | Per episode (on reset) |
| **Random Network Adversary** | | |
| RNA architecture | 512→512→1024→1024→out×32 | 512→512→1024→1024→out×32 |
| RNA output | softmax bins → weighted sum | softmax bins → weighted sum |
| RNA blending | `α × rna + (1-α) × action` | `α × rna + (1-α) × action` |
| ADR-controlled RNA alpha | Yes | Yes |
| **Random Pose Injection** | | |
| Injection method | `Bernoulli(p)` per step | `Bernoulli(p)` per step |
| Probability sampling | `p ~ U(0, 0.3)` per episode | `p ~ U(0, 0.3)` per episode |
| Random pose | `init_pos ± 0.5` + random quat | `init_pos ± 0.5` + random quat |
| Pipeline position | After affine noise, before refresh gate | After affine noise, before refresh gate |

### 주요 차이점 (의도적 변경)

1. **시뮬레이터**: IsaacGym (PhysX) → Newton (MuJoCo solver)
2. **RL 라이브러리**: rl_games → 직접 구현 (알고리즘 동일)
3. **Observation**: 50 → 72 dims (dof_vel 16 + object_vels 6 추가)
4. **Coordinate frame**: wrist-relative → hand-base-relative (간소화)

## 참고 자료

- [IsaacGymEnvs DextrEme](https://github.com/isaac-sim/IsaacGymEnvs/tree/main/isaacgymenvs/tasks/dextreme)
- [rl_games](https://github.com/Denys88/rl_games) - PPO 알고리즘 참조
- [IsaacLab Allegro Hand](https://github.com/isaac-sim/IsaacLab/blob/main/source/isaaclab_tasks/isaaclab_tasks/direct/allegro_hand/)
- [Newton Examples](https://github.com/nvidia-warp/newton)
