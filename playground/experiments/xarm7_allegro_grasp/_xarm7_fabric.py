"""xArm7-specific FABRICS action controller.

Subclasses :class:`FabricActionController` and replaces only the modified-DH
constants used by ``compute_franka_fk`` / ``compute_franka_jacobian``. The
field names ``dh_a``, ``dh_d``, ``dh_alpha`` are reused (the parent's FK loop
indexes them by name) — only their values change. ``GraspFabric`` doesn't
touch these constants and is reused unchanged.

xArm7 modified DH (Craig convention), in metres / radians:

    i  | a_{i-1} | d_i    | alpha_{i-1}
    ---|---------|--------|------------
    1  | 0       | 0.267  | 0
    2  | 0       | 0      | -pi/2
    3  | 0.0525  | 0      |  pi/2
    4  | 0.0775  | 0.293  |  pi/2
    5  | 0       | 0      |  pi/2
    6  | 0.076   | 0      |  pi/2
    7  | 0       | 0.097  | -pi/2

Indexing follows the parent ``FabricActionController``: 8 entries per array
(7 joints + a trailing flange step with all zeros) so the inner FK loop
``for i in range(8)`` iterates correctly.
"""

from __future__ import annotations

import math

import torch

from playground.experiments.franka_allegro_grasp.fabric import FabricActionController


class XArm7FabricActionController(FabricActionController):
    """FABRICS controller with UFactory xArm7 kinematics."""

    def _init_franka_kinematics(self):
        # a_{i-1} for joints 1..7, then flange (zero).
        self.register_buffer(
            "dh_a",
            torch.tensor(
                [0.0, 0.0, 0.0525, 0.0775, 0.0, 0.076, 0.0, 0.0],
                dtype=torch.float32,
                device=self.device,
            ),
        )
        # d_i for joints 1..7, then flange (zero).
        self.register_buffer(
            "dh_d",
            torch.tensor(
                [0.267, 0.0, 0.0, 0.293, 0.0, 0.0, 0.097, 0.0],
                dtype=torch.float32,
                device=self.device,
            ),
        )
        # alpha_{i-1} for joints 1..7, then flange (zero).
        pi = math.pi
        self.register_buffer(
            "dh_alpha",
            torch.tensor(
                [0.0, -pi / 2, pi / 2, pi / 2, pi / 2, pi / 2, -pi / 2, 0.0],
                dtype=torch.float32,
                device=self.device,
            ),
        )
