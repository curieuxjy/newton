"""Allegro Hand Cube Rotation with PPO - Newton RL Example.

This package implements a PPO-based RL training for cube manipulation
using the Allegro Hand, similar to IsaacLab's DextrEme but using only Newton.

Usage:
    # Train
    python -m playground.experiments.allegro_cube_ppo.train --num-envs 4096

    # Or import directly
    from playground.experiments.allegro_cube_ppo import AllegroHandCubeEnv, PPO, train
"""

from .adr import ADRManager, ADRParamDef, make_default_adr_params
from .config import DRConfig, EnvConfig, PPOConfig, TrainConfig, WandbConfig
from .env import AllegroHandCubeEnv
from .ppo import ActorCritic, PPO, RolloutBuffer
from .train import train

__all__ = [
    "ADRManager",
    "ADRParamDef",
    "AllegroHandCubeEnv",
    "ActorCritic",
    "DRConfig",
    "PPO",
    "RolloutBuffer",
    "EnvConfig",
    "PPOConfig",
    "TrainConfig",
    "WandbConfig",
    "make_default_adr_params",
    "train",
]
