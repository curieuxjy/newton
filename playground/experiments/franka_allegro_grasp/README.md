# Franka + Allegro Hand Cube Grasping (DEXTRAH-G)

[DEXTRAH-G](https://github.com/NVlabs/DEXTRAH)의 2단계 teacher-student distillation 파이프라인 구현.
로봇 구조, DOF 매핑 등 기본 정보는 `../franka_allegro/README.md` 참조.

## 실행

```bash
# Teacher 학습
uv run --extra examples --extra torch-cu12 python -m playground.experiments.franka_allegro_grasp.train \
    --num-envs 256 --no-wandb

# Student distillation
uv run --extra examples --extra torch-cu12 python -m playground.experiments.franka_allegro_grasp.distill \
    --teacher-checkpoint checkpoints/teacher_*/best.pt

# 시각화
uv run --extra examples --extra torch-cu12 --with matplotlib \
    python -m playground.experiments.franka_allegro_grasp.visualize

# TensorBoard
tensorboard --logdir runs/
```

아키텍처, observation, reward, 하이퍼파라미터 등 상세 문서는 `../franka_allegro/README.md`의 DEXTRAH-G 섹션을 참조하세요.
