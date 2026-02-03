"""Allegro Hand Cube Rotation with PPO - Newton RL Example.

This package implements a PPO-based RL training for cube manipulation
using the Allegro Hand, similar to IsaacLab's DextrEme but using only Newton.

Usage:
    # Train
    python -m playground.experiments.allegro_cube_ppo.train --num-envs 4096

    # Or import directly
    from playground.experiments.allegro_cube_ppo import AllegroHandCubeEnv, PPO, train
"""

from .config import EnvConfig, PPOConfig, TrainConfig, WandbConfig
from .env import AllegroHandCubeEnv
from .ppo import ActorCritic, PPO, RolloutBuffer
from .train import train

__all__ = [
    "AllegroHandCubeEnv",
    "ActorCritic",
    "PPO",
    "RolloutBuffer",
    "EnvConfig",
    "PPOConfig",
    "TrainConfig",
    "WandbConfig",
    "train",
]
