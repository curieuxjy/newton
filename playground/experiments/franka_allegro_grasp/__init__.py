"""Franka + Allegro hand cube grasping example with depth sensor and FABRICS.

DEXTRAH-G two-stage training pipeline:
  Stage 1: Train privileged teacher with LSTM PPO (train.py)
  Stage 2: Distill teacher into student depth policy (distill.py)
"""

from .config import (
    DistillConfig,
    EnvConfig,
    PPOConfig,
    StudentConfig,
    TeacherPPOConfig,
    TrainConfig,
)
from .env import FrankaAllegroGraspEnv
from .fabric import (
    FabricActionController,
    FingertipTaskMap,
    GraspFabric,
    HAND_PCA_MATRIX,
    HAND_PCA_MAXS,
    HAND_PCA_MINS,
    LinearTaskMap,
    PALM_POSE_MAXS,
    PALM_POSE_MINS,
    TaskMap,
)
from .networks import ObsRunningMeanStd, StudentNetwork, TeacherActorCritic

__all__ = [
    "FrankaAllegroGraspEnv",
    "EnvConfig",
    "PPOConfig",
    "TeacherPPOConfig",
    "StudentConfig",
    "DistillConfig",
    "TrainConfig",
    "FabricActionController",
    "GraspFabric",
    "TaskMap",
    "LinearTaskMap",
    "FingertipTaskMap",
    "HAND_PCA_MATRIX",
    "HAND_PCA_MINS",
    "HAND_PCA_MAXS",
    "PALM_POSE_MINS",
    "PALM_POSE_MAXS",
    "TeacherActorCritic",
    "StudentNetwork",
]
