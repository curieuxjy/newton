# Allegro Hand Cube Rotation with PPO

Newton 물리 엔진을 사용한 Allegro Hand 큐브 회전 강화학습 예제입니다.
IsaacLab의 DextrEme를 IsaacSim 없이 Newton만으로 구현한 프로젝트입니다.

## 프로젝트 구조

```
allegro_cube_ppo/
├── __init__.py      # 패키지 export
├── config.py        # 하이퍼파라미터 설정
├── env.py           # AllegroHandCubeEnv (Newton 기반 환경)
├── ppo.py           # PPO 알고리즘 (CleanRL 스타일)
├── train.py         # 학습 스크립트
├── visualize.py     # Newton viewer로 시각화
└── README.md        # 이 파일
```

## 환경 설명

### Observation (43 dims)
| 항목 | 차원 | 설명 |
|------|------|------|
| Joint positions | 16 | 손가락 관절 위치 |
| Joint velocities | 16 | 손가락 관절 속도 |
| Cube relative position | 3 | 손바닥 기준 큐브 위치 |
| Cube orientation | 4 | 큐브 쿼터니언 |
| Goal orientation | 4 | 목표 쿼터니언 |

### Action (16 dims)
- 16개 손가락 관절의 position target (PD control)
- 범위: [-1, 1] → joint limits로 스케일링

### Reward
- **Rotation alignment**: 큐브와 목표 회전의 쿼터니언 거리
- **Action penalty**: 액션 크기 페널티
- **Action rate penalty**: 액션 변화율 페널티

### Termination
- Episode timeout (기본 600 steps = 12초)
- 큐브 낙하 (z < 0.05m)

## 실행 방법

### 1. 시각화 (Newton Viewer)

학습 없이 시뮬레이터 확인:
```bash
uv run --extra examples --extra torch-cu12 python -m playground.experiments.allegro_cube_ppo.visualize
```

학습된 정책으로 시각화:
```bash
uv run --extra examples --extra torch-cu12 python -m playground.experiments.allegro_cube_ppo.visualize \
    --checkpoint checkpoints/allegro_cube_ppo_XXXXXX/final.pt
```

여러 환경 동시 시각화:
```bash
uv run --extra examples --extra torch-cu12 python -m playground.experiments.allegro_cube_ppo.visualize \
    --num-envs 16
```

### 2. 학습

기본 학습:
```bash
uv run --extra examples --extra torch-cu12 python -m playground.experiments.allegro_cube_ppo.train \
    --num-envs 4096 \
    --total-timesteps 100000000
```

빠른 테스트 (적은 환경):
```bash
uv run --extra examples --extra torch-cu12 python -m playground.experiments.allegro_cube_ppo.train \
    --num-envs 256 \
    --total-timesteps 1000000 \
    --no-wandb
```

### 3. 로깅 옵션

#### TensorBoard (기본 활성화)
```bash
# 학습 실행 후 별도 터미널에서:
tensorboard --logdir runs/
# 브라우저에서 http://localhost:6006 접속
```

#### Wandb (선택)
```bash
# 설치
pip install wandb
wandb login

# wandb와 함께 학습
uv run --extra examples --extra torch-cu12 python -m playground.experiments.allegro_cube_ppo.train \
    --num-envs 4096 \
    --wandb-project allegro-cube-ppo \
    --wandb-entity YOUR_USERNAME
```

## CLI 인자

### train.py
| 인자 | 기본값 | 설명 |
|------|--------|------|
| `--num-envs` | 4096 | 병렬 환경 수 |
| `--total-timesteps` | 100M | 총 학습 스텝 |
| `--seed` | 42 | 랜덤 시드 |
| `--device` | cuda | 디바이스 (cuda/cpu) |
| `--checkpoint-dir` | checkpoints | 체크포인트 저장 경로 |
| `--wandb` / `--no-wandb` | 활성화 | wandb 로깅 on/off |
| `--wandb-project` | allegro-cube-ppo | wandb 프로젝트명 |
| `--wandb-entity` | None | wandb 사용자/팀명 |

### visualize.py
| 인자 | 기본값 | 설명 |
|------|--------|------|
| `--checkpoint` | None | 학습된 체크포인트 경로 |
| `--num-envs` | 4 | 시각화할 환경 수 |

## 하이퍼파라미터

### EnvConfig (config.py)
```python
num_envs: int = 4096
episode_length: int = 600      # 12초 @ 50Hz
fps: int = 50
sim_substeps: int = 4
control_decimation: int = 2    # 25Hz 제어
hand_stiffness: float = 150.0
hand_damping: float = 5.0
```

### PPOConfig (config.py)
```python
learning_rate: float = 3e-4
gamma: float = 0.99
gae_lambda: float = 0.95
clip_epsilon: float = 0.2
entropy_coef: float = 0.01
num_epochs: int = 5
num_minibatches: int = 4
rollout_steps: int = 16
hidden_dims: tuple = (256, 256, 128)
```

## 체크포인트 구조

```python
{
    "actor_critic": model.state_dict(),
    "optimizer": optimizer.state_dict(),
}
```

## 확장 아이디어

- [ ] Fingertip 접촉 보상 추가
- [ ] Domain randomization (물리 파라미터)
- [ ] Asymmetric actor-critic (privileged info)
- [ ] 다른 오브젝트 (구, 실린더 등)
- [ ] Right hand 버전
