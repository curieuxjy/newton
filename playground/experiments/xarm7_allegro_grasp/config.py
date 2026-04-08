"""Configuration for xArm7 + Allegro cube grasping environment.

Re-uses :mod:`playground.experiments.franka_allegro_grasp.config` since the
RL/distillation/wandb settings are arm-agnostic. Only environment defaults
that depend on the arm choice are overridden via :func:`make_xarm7_env_config`.

Note: the field names in :class:`EnvConfig` still read ``franka_*`` for the
arm because the parent class uses those names internally. They are interpreted
as "arm" parameters here.
"""

from __future__ import annotations

from playground.experiments.franka_allegro_grasp.config import (
    DistillConfig,
    EnvConfig,
    PPOConfig,
    StudentConfig,
    TeacherPPOConfig,
    TrainConfig,
    WandbConfig,
)


def make_xarm7_env_config(**overrides) -> EnvConfig:
    """Return an :class:`EnvConfig` with xArm7-tuned arm gains.

    UFactory's default xArm7 ROS controller uses higher position stiffness
    and lower torque limits than Franka FR3. The values below are reasonable
    starting points; adjust per your task.
    """
    # Place the cube just above the table top.
    # Table center z = table_height/2 = 0.20, half-thickness = 0.01,
    # so the top surface is at z ≈ 0.21. Cube half-extent = cube_size/2 = 0.025,
    # so the cube rests on the table at z ≈ 0.235.
    _table_top_z = 0.4 / 2 + 0.02 / 2
    _cube_rest_z = _table_top_z + 0.05 / 2 + 1e-3

    defaults = dict(
        # The ``franka_*`` field names refer to the arm in general — here xArm7.
        franka_stiffness=800.0,
        franka_damping=40.0,
        franka_effort_limit=50.0,
        franka_armature=0.1,
        cube_spawn_pos=(-0.3, -0.5, _cube_rest_z),
    )
    defaults.update(overrides)
    return EnvConfig(**defaults)


def make_xarm7_train_config(**overrides) -> TrainConfig:
    """Return a :class:`TrainConfig` pre-named for the xArm7 experiment."""
    train_defaults = dict(
        experiment_name="xarm7_allegro_grasp",
        env=make_xarm7_env_config(),
    )
    train_defaults.update(overrides)
    cfg = TrainConfig(**train_defaults)
    cfg.wandb.project = "newton-xarm7-allegro-grasp"
    cfg.wandb.tags = ["ppo", "xarm7", "allegro", "grasp", "newton"]
    return cfg


__all__ = [
    "DistillConfig",
    "EnvConfig",
    "PPOConfig",
    "StudentConfig",
    "TeacherPPOConfig",
    "TrainConfig",
    "WandbConfig",
    "make_xarm7_env_config",
    "make_xarm7_train_config",
]
