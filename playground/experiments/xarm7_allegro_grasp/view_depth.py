"""Depth-image viewer entry point for xArm7 + Allegro grasp.

Run with:
    uv run --extra examples --extra torch-cu12 \
        python -m playground.experiments.xarm7_allegro_grasp.view_depth
"""

from __future__ import annotations

from playground.experiments.franka_allegro_grasp import view_depth as _franka_view_depth

from .config import make_xarm7_env_config
from .env import XArm7AllegroGraspEnv

_franka_view_depth.FrankaAllegroGraspEnv = XArm7AllegroGraspEnv
_franka_view_depth.EnvConfig = lambda **kwargs: make_xarm7_env_config(**kwargs)


def main():
    _franka_view_depth.main()


if __name__ == "__main__":
    main()
