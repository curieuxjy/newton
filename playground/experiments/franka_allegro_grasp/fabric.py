"""Simplified FABRICS implementation for Franka + Allegro grasping.

Reference: NVlabs/FABRICS - Riemannian Geometric Fabrics for Robot Motion Planning
https://github.com/NVlabs/FABRICS

This is a simplified version that provides:
1. Task space position computation (fingertips, palm)
2. Jacobian-based velocity mapping
3. Multi-point attractor targets for grasp control
"""

import torch
import torch.nn as nn
import numpy as np


class TaskMap(nn.Module):
    """Base class for task maps that transform joint space to task space."""

    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim

    def forward(self, q: torch.Tensor) -> torch.Tensor:
        """Compute task space position from joint configuration.

        Args:
            q: Joint positions (batch_size, input_dim)

        Returns:
            x: Task space positions (batch_size, output_dim)
        """
        raise NotImplementedError


class LinearTaskMap(TaskMap):
    """Linear task map: x = A @ q + b

    Used for PCA-based hand control where A is the PCA projection matrix.
    """

    def __init__(self, A: torch.Tensor, b: torch.Tensor | None = None):
        """
        Args:
            A: Projection matrix (output_dim, input_dim)
            b: Bias vector (output_dim,)
        """
        super().__init__(A.shape[1], A.shape[0])
        self.register_buffer("A", A)
        if b is not None:
            self.register_buffer("b", b)
        else:
            self.register_buffer("b", torch.zeros(A.shape[0]))

    def forward(self, q: torch.Tensor) -> torch.Tensor:
        # q: (batch, input_dim) -> x: (batch, output_dim)
        return torch.matmul(q, self.A.T) + self.b

    def jacobian(self, q: torch.Tensor) -> torch.Tensor:
        """Return Jacobian (constant for linear map)."""
        batch_size = q.shape[0]
        return self.A.unsqueeze(0).expand(batch_size, -1, -1)


class IdentityTaskMap(TaskMap):
    """Identity task map for joint space control."""

    def __init__(self, dim: int):
        super().__init__(dim, dim)

    def forward(self, q: torch.Tensor) -> torch.Tensor:
        return q

    def jacobian(self, q: torch.Tensor) -> torch.Tensor:
        batch_size = q.shape[0]
        return torch.eye(self.input_dim, device=q.device).unsqueeze(0).expand(batch_size, -1, -1)


class FingertipTaskMap(TaskMap):
    """Task map for computing fingertip positions from joint angles.

    For Allegro hand, this computes approximate fingertip positions
    based on forward kinematics approximation.
    """

    def __init__(
        self,
        num_fingers: int = 4,
        joints_per_finger: int = 4,
        link_lengths: list[float] | None = None,
        device: str = "cuda",
    ):
        output_dim = num_fingers * 3  # 3D position per fingertip
        input_dim = num_fingers * joints_per_finger
        super().__init__(input_dim, output_dim)

        self.num_fingers = num_fingers
        self.joints_per_finger = joints_per_finger

        # Default link lengths for Allegro hand (approximate)
        if link_lengths is None:
            link_lengths = [0.054, 0.038, 0.044, 0.027]  # proximal to distal
        self.register_buffer("link_lengths", torch.tensor(link_lengths, device=device))

        # Finger base positions (relative to palm center, approximate)
        finger_bases = torch.tensor([
            [0.0, 0.0335, 0.029],   # Index (link_0)
            [0.0, 0.0, 0.029],      # Middle (link_4)
            [0.0, -0.0335, 0.029],  # Ring (link_8)
            [-0.027, 0.018, 0.022], # Thumb (link_12)
        ], device=device)
        self.register_buffer("finger_bases", finger_bases)

    def forward(self, q: torch.Tensor) -> torch.Tensor:
        """Compute fingertip positions (simplified FK).

        Args:
            q: Allegro joint positions (batch, 16)

        Returns:
            fingertip_pos: Fingertip positions (batch, 12) - 4 fingers * 3D
        """
        batch_size = q.shape[0]
        device = q.device

        fingertips = []
        for f in range(self.num_fingers):
            start_idx = f * self.joints_per_finger
            finger_q = q[:, start_idx:start_idx + self.joints_per_finger]

            # Simple planar FK approximation
            # Joint 0: abduction (side-to-side)
            # Joints 1-3: flexion (bending)

            base = self.finger_bases[f].unsqueeze(0).expand(batch_size, -1)

            # Cumulative angle for flexion joints
            cumsum_angles = torch.cumsum(finger_q[:, 1:], dim=1)

            # Fingertip position in finger frame
            x = torch.zeros(batch_size, device=device)
            y = torch.zeros(batch_size, device=device)
            z = self.link_lengths[0].clone()

            for i in range(3):
                angle = cumsum_angles[:, i]
                length = self.link_lengths[i + 1] if i < 3 else 0
                z = z + length * torch.cos(angle)
                y = y + length * torch.sin(angle)

            # Apply abduction
            abduct = finger_q[:, 0]
            x_rot = x * torch.cos(abduct) - y * torch.sin(abduct)
            y_rot = x * torch.sin(abduct) + y * torch.cos(abduct)

            fingertip = base + torch.stack([x_rot, y_rot, z], dim=1)
            fingertips.append(fingertip)

        return torch.cat(fingertips, dim=1)


class GraspFabric(nn.Module):
    """Simplified FABRICS controller for grasping tasks.

    Combines multiple task maps:
    1. Palm position control (reach)
    2. Fingertip position control (grasp)
    3. Joint space regularization
    """

    def __init__(
        self,
        franka_dof: int = 7,
        allegro_dof: int = 16,
        device: str = "cuda",
    ):
        super().__init__()
        self.franka_dof = franka_dof
        self.allegro_dof = allegro_dof
        self.total_dof = franka_dof + allegro_dof
        self.device = device

        # Task maps
        self.fingertip_taskmap = FingertipTaskMap(device=device)

        # Hand PCA (simplified - use first 5 principal components)
        # This allows 5D control of the 16D hand configuration
        self._init_hand_pca()

        # Attractor gains (learnable or fixed)
        self.palm_gain = nn.Parameter(torch.tensor(10.0))
        self.fingertip_gain = nn.Parameter(torch.tensor(5.0))
        self.joint_reg_gain = nn.Parameter(torch.tensor(0.1))

        # Damping
        self.damping_gain = nn.Parameter(torch.tensor(1.0))

    def _init_hand_pca(self):
        """Initialize PCA matrix for hand control.

        A simplified version using predefined synergies:
        - PC1: Power grasp (all fingers close)
        - PC2: Precision grasp (thumb + index)
        - PC3: Index point
        - PC4: Spread fingers
        - PC5: Thumb opposition
        """
        # Simplified PCA-like projection (5D -> 16D)
        pca_matrix = torch.zeros(5, 16, device=self.device)

        # PC1: All fingers close equally
        pca_matrix[0, :] = 1.0
        pca_matrix[0, 0::4] = 0.0  # Skip abduction joints

        # PC2: Thumb + index close
        pca_matrix[1, 8:12] = 1.0  # Index (DOF 8-11)
        pca_matrix[1, 12:16] = 1.0  # Thumb (DOF 12-15)
        pca_matrix[1, 8] = 0.0  # Skip abduction
        pca_matrix[1, 12] = 0.0

        # PC3: Index point (extend index, close others)
        pca_matrix[2, 0:8] = 1.0  # Ring + Middle close
        pca_matrix[2, 8:12] = -0.5  # Index extend
        pca_matrix[2, 0::4] = 0.0

        # PC4: Spread fingers (abduction)
        pca_matrix[3, 0::4] = 1.0

        # PC5: Thumb opposition
        pca_matrix[4, 12:16] = 1.0

        # Normalize rows
        pca_matrix = pca_matrix / (pca_matrix.norm(dim=1, keepdim=True) + 1e-6)

        self.register_buffer("hand_pca", pca_matrix)
        self.hand_taskmap = LinearTaskMap(pca_matrix)

    def compute_fingertip_positions(
        self,
        allegro_q: torch.Tensor,
        ee_pos: torch.Tensor,
        ee_quat: torch.Tensor,
    ) -> torch.Tensor:
        """Compute world-frame fingertip positions.

        Args:
            allegro_q: Hand joint positions (batch, 16)
            ee_pos: End effector position (batch, 3)
            ee_quat: End effector quaternion (batch, 4) - (x, y, z, w)

        Returns:
            fingertip_pos: World-frame fingertip positions (batch, 4, 3)
        """
        # Get fingertip positions in hand frame
        fingertips_local = self.fingertip_taskmap(allegro_q)  # (batch, 12)
        fingertips_local = fingertips_local.view(-1, 4, 3)  # (batch, 4, 3)

        # Transform to world frame
        fingertips_world = self._transform_points(fingertips_local, ee_pos, ee_quat)

        return fingertips_world

    def _transform_points(
        self,
        points: torch.Tensor,
        pos: torch.Tensor,
        quat: torch.Tensor,
    ) -> torch.Tensor:
        """Transform points from local to world frame.

        Args:
            points: Local points (batch, N, 3)
            pos: Translation (batch, 3)
            quat: Rotation quaternion (batch, 4) - (x, y, z, w)

        Returns:
            world_points: Transformed points (batch, N, 3)
        """
        batch_size = points.shape[0]
        n_points = points.shape[1]

        # Quaternion to rotation matrix
        R = self._quat_to_rotation_matrix(quat)  # (batch, 3, 3)

        # Apply rotation
        points_flat = points.view(batch_size, n_points, 3, 1)
        R_expanded = R.unsqueeze(1).expand(-1, n_points, -1, -1)
        rotated = torch.matmul(R_expanded, points_flat).squeeze(-1)

        # Apply translation
        pos_expanded = pos.unsqueeze(1).expand(-1, n_points, -1)
        world_points = rotated + pos_expanded

        return world_points

    def _quat_to_rotation_matrix(self, quat: torch.Tensor) -> torch.Tensor:
        """Convert quaternion to rotation matrix.

        Args:
            quat: Quaternion (batch, 4) - (x, y, z, w)

        Returns:
            R: Rotation matrix (batch, 3, 3)
        """
        x, y, z, w = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]

        R = torch.zeros(quat.shape[0], 3, 3, device=quat.device)
        R[:, 0, 0] = 1 - 2 * (y * y + z * z)
        R[:, 0, 1] = 2 * (x * y - z * w)
        R[:, 0, 2] = 2 * (x * z + y * w)
        R[:, 1, 0] = 2 * (x * y + z * w)
        R[:, 1, 1] = 1 - 2 * (x * x + z * z)
        R[:, 1, 2] = 2 * (y * z - x * w)
        R[:, 2, 0] = 2 * (x * z - y * w)
        R[:, 2, 1] = 2 * (y * z + x * w)
        R[:, 2, 2] = 1 - 2 * (x * x + y * y)

        return R

    def compute_grasp_features(
        self,
        ee_pos: torch.Tensor,
        ee_quat: torch.Tensor,
        allegro_q: torch.Tensor,
        cube_pos: torch.Tensor,
        cube_size: float = 0.05,
    ) -> dict[str, torch.Tensor]:
        """Compute grasp-relevant features for observations/rewards.

        Args:
            ee_pos: End effector position (batch, 3)
            ee_quat: End effector quaternion (batch, 4)
            allegro_q: Allegro joint positions (batch, 16)
            cube_pos: Cube center position (batch, 3)
            cube_size: Cube side length

        Returns:
            Dictionary with:
            - fingertip_positions: (batch, 4, 3)
            - fingertip_to_cube: (batch, 4, 3) distances to cube center
            - grasp_closure: (batch,) how well fingers surround cube
            - palm_to_cube: (batch, 3) palm center to cube
        """
        # Compute fingertip positions
        fingertips = self.compute_fingertip_positions(allegro_q, ee_pos, ee_quat)

        # Fingertip to cube vectors
        cube_expanded = cube_pos.unsqueeze(1).expand(-1, 4, -1)
        fingertip_to_cube = fingertips - cube_expanded

        # Grasp closure metric: how well fingers surround the object
        # Compute centroid of fingertips and compare to cube center
        fingertip_centroid = fingertips.mean(dim=1)  # (batch, 3)
        centroid_to_cube = cube_pos - fingertip_centroid

        # Closure score: small when fingertip centroid is near cube
        grasp_closure = centroid_to_cube.norm(dim=-1)

        # Palm to cube
        palm_to_cube = cube_pos - ee_pos

        return {
            "fingertip_positions": fingertips,
            "fingertip_to_cube": fingertip_to_cube,
            "fingertip_distances": fingertip_to_cube.norm(dim=-1),  # (batch, 4)
            "grasp_closure": grasp_closure,
            "palm_to_cube": palm_to_cube,
        }

    def compute_grasp_reward(
        self,
        grasp_features: dict[str, torch.Tensor],
        cube_size: float = 0.05,
    ) -> dict[str, torch.Tensor]:
        """Compute grasp-based reward components.

        Args:
            grasp_features: Output from compute_grasp_features
            cube_size: Cube side length

        Returns:
            Dictionary with reward components
        """
        fingertip_dist = grasp_features["fingertip_distances"]  # (batch, 4)
        closure = grasp_features["grasp_closure"]  # (batch,)
        palm_to_cube = grasp_features["palm_to_cube"]  # (batch, 3)

        # Target fingertip distance (touching cube surface)
        target_dist = cube_size / 2 + 0.01  # slight offset

        # Fingertip contact reward: encourage fingertips to be at target distance
        fingertip_error = (fingertip_dist - target_dist).abs()
        contact_reward = torch.exp(-10.0 * fingertip_error.mean(dim=1))

        # Closure reward: encourage centroid to be at cube center
        closure_reward = torch.exp(-5.0 * closure)

        # Approach reward: palm close to cube
        approach_dist = palm_to_cube.norm(dim=-1)
        approach_reward = torch.exp(-3.0 * approach_dist)

        # Multi-finger contact bonus: reward when multiple fingers are close
        close_fingers = (fingertip_dist < target_dist * 1.5).float().sum(dim=1)
        multi_contact_bonus = close_fingers / 4.0

        return {
            "contact_reward": contact_reward,
            "closure_reward": closure_reward,
            "approach_reward": approach_reward,
            "multi_contact_bonus": multi_contact_bonus,
        }

    def forward(
        self,
        franka_q: torch.Tensor,
        franka_qd: torch.Tensor,
        allegro_q: torch.Tensor,
        allegro_qd: torch.Tensor,
        ee_pos: torch.Tensor,
        ee_quat: torch.Tensor,
        target_pos: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute fabric-based control forces.

        This is a simplified version that computes attractor forces
        rather than the full Riemannian metric formulation.

        Args:
            franka_q: Arm joint positions (batch, 7)
            franka_qd: Arm joint velocities (batch, 7)
            allegro_q: Hand joint positions (batch, 16)
            allegro_qd: Hand joint velocities (batch, 16)
            ee_pos: End effector position (batch, 3)
            ee_quat: End effector quaternion (batch, 4)
            target_pos: Target position for EE (batch, 3)

        Returns:
            franka_force: Arm control signal (batch, 7)
            allegro_force: Hand control signal (batch, 16)
        """
        batch_size = franka_q.shape[0]

        # Palm position error
        palm_error = target_pos - ee_pos

        # Simple proportional control for arm (move EE to target)
        # In full FABRICS, this would be proper Jacobian transpose control
        franka_force = torch.zeros(batch_size, self.franka_dof, device=self.device)
        franka_force[:, :3] = self.palm_gain * palm_error  # Simplified

        # Add damping
        franka_force = franka_force - self.damping_gain * franka_qd

        # Hand control via PCA space
        # Get current PCA coordinates
        hand_pca_coords = self.hand_taskmap(allegro_q)  # (batch, 5)

        # Target: closed grasp (high PC1)
        target_pca = torch.zeros(batch_size, 5, device=self.device)
        target_pca[:, 0] = 1.0  # Power grasp

        pca_error = target_pca - hand_pca_coords

        # Map back to joint space
        allegro_force = torch.matmul(pca_error, self.hand_pca)
        allegro_force = self.fingertip_gain * allegro_force

        # Add damping
        allegro_force = allegro_force - self.damping_gain * allegro_qd

        return franka_force, allegro_force
