"""Teacher PPO training entry point for xArm7 + Allegro grasp.

Run with:
    uv run --extra examples --extra torch-cu12 \
        python -m playground.experiments.xarm7_allegro_grasp.train

This thin wrapper reuses the franka_allegro_grasp training loop unchanged
and swaps the environment / config classes for the xArm7 variants.
"""

from __future__ import annotations

from playground.experiments.franka_allegro_grasp import train as _franka_train

from .config import make_xarm7_env_config
from .env import XArm7AllegroGraspEnv

# Monkey-patch the franka training module so its ``main()`` instantiates the
# xArm7 environment and reads xArm7-tuned defaults from the config factory.
_franka_train.FrankaAllegroGraspEnv = XArm7AllegroGraspEnv
_franka_train.EnvConfig = lambda **kwargs: make_xarm7_env_config(**kwargs)


def main():
    _franka_train.main()


if __name__ == "__main__":
    main()
