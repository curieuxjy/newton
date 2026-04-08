"""Tiny mujoco_menagerie download helper for playground experiments.

Newton's public API only exposes ``newton.utils.download_asset`` which fetches
from the ``newton-assets`` repository. xArm7 lives in ``mujoco_menagerie``, so
we vendor a small git-clone helper here. Pinned to the same revision Newton's
test suite uses.

Set ``NEWTON_MENAGERIE_PATH`` to point at an existing clone to skip download.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

MENAGERIE_URL = "https://github.com/google-deepmind/mujoco_menagerie.git"
MENAGERIE_REF = "feadf76d42f8a2162426f7d226a3b539556b3bf5"

NEWTON_MENAGERIE_PATH_ENV = "NEWTON_MENAGERIE_PATH"


def _default_cache_dir() -> Path:
    base = os.environ.get("XDG_CACHE_HOME") or str(Path.home() / ".cache")
    return Path(base) / "newton" / f"mujoco_menagerie_{MENAGERIE_REF[:8]}"


def download_menagerie_asset(robot_folder: str, force_refresh: bool = False) -> Path:
    """Return a local path to ``<menagerie>/<robot_folder>``.

    If ``NEWTON_MENAGERIE_PATH`` is set and contains the folder, use it.
    Otherwise clone the pinned revision into the user cache directory.
    """
    local_root = os.environ.get(NEWTON_MENAGERIE_PATH_ENV)
    if local_root and not force_refresh:
        path = Path(local_root) / robot_folder
        if path.exists():
            return path

    cache_dir = _default_cache_dir()
    if force_refresh and cache_dir.exists():
        import shutil

        shutil.rmtree(cache_dir)

    if not (cache_dir / ".git").exists():
        cache_dir.parent.mkdir(parents=True, exist_ok=True)
        print(f"[xarm7_allegro_grasp] Cloning mujoco_menagerie @ {MENAGERIE_REF[:8]} ...")
        subprocess.run(
            ["git", "clone", "--filter=blob:none", MENAGERIE_URL, str(cache_dir)],
            check=True,
        )
        subprocess.run(
            ["git", "-C", str(cache_dir), "checkout", "--quiet", MENAGERIE_REF],
            check=True,
        )

    asset_path = cache_dir / robot_folder
    if not asset_path.exists():
        raise FileNotFoundError(
            f"'{robot_folder}' not found in mujoco_menagerie clone at {cache_dir}"
        )
    return asset_path
