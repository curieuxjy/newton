"""Visualization entry point for xArm7 + Allegro grasp.

Run with:
    uv run --extra examples --extra torch-cu12 \
        python -m playground.experiments.xarm7_allegro_grasp.visualize
"""

from __future__ import annotations

from playground.experiments.franka_allegro_grasp import visualize as _franka_visualize

from .config import make_xarm7_env_config
from .env import XArm7AllegroGraspEnv

_franka_visualize.FrankaAllegroGraspEnv = XArm7AllegroGraspEnv
_franka_visualize.EnvConfig = lambda **kwargs: make_xarm7_env_config(**kwargs)


# The franka visualize module exposes a ``VisualizeExample`` class and is
# launched via ``newton.examples.run`` from its ``__main__`` block. We mirror
# that here so ``python -m ...visualize`` behaves identically.
VisualizeExample = _franka_visualize.VisualizeExample


if __name__ == "__main__":
    _franka_visualize.main()
