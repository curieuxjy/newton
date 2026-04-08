"""xArm7 + Allegro hand cube grasping example.

Variant of :mod:`playground.experiments.franka_allegro_grasp` that swaps the
Franka FR3 arm for a UFactory xArm7 (loaded from mujoco_menagerie). All RL
training, distillation, FABRICS, and network code is reused unchanged from
the franka_allegro_grasp package.
"""

from playground.experiments.franka_allegro_grasp import (
    DistillConfig,
    PPOConfig,
    StudentConfig,
    TeacherPPOConfig,
    TrainConfig,
)
from playground.experiments.franka_allegro_grasp.fabric import (
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
from playground.experiments.franka_allegro_grasp.networks import (
    ObsRunningMeanStd,
    StudentNetwork,
    TeacherActorCritic,
)

from .config import EnvConfig, make_xarm7_env_config, make_xarm7_train_config
from .env import XArm7AllegroGraspEnv

__all__ = [
    "XArm7AllegroGraspEnv",
    "EnvConfig",
    "make_xarm7_env_config",
    "make_xarm7_train_config",
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
    "ObsRunningMeanStd",
]
