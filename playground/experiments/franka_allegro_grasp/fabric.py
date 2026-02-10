"""FABRICS implementation for Franka + Allegro grasping.

Reference:
- NVlabs/FABRICS - Riemannian Geometric Fabrics for Robot Motion Planning
  https://github.com/NVlabs/FABRICS
- NVlabs/DEXTRAH - High-performance hand-arm grasping policy
  https://github.com/NVlabs/DEXTRAH

This module provides:
1. FabricActionController: Converts 11D fabric actions to 23D joint targets
   - 6D palm pose (XYZ + RPY) -> 7D Franka arm joints via IK
   - 5D hand PCA -> 16D Allegro hand joints via PCA matrix
2. GraspFabric: Computes grasp features and rewards
"""

import torch
import torch.nn as nn
import numpy as np


# === DEXTRAH Constants ===
# Reference: dextrah_lab/tasks/dextrah_kuka_allegro/dextrah_kuka_allegro_constants.py

# Hand PCA limits (5D action space for 16D Allegro hand)
HAND_PCA_MINS = [0.2475, -0.3286, -0.7238, -0.0192, -0.5532]
HAND_PCA_MAXS = [3.8336, 3.0025, 0.8977, 1.0243, 0.0629]

# Palm pose limits (will be adjusted for Franka's workspace)
# Original DEXTRAH (for Kuka): position [-1.2, -0.7, 0.0] to [0.0, 0.7, 1.0]
# Adjusted for Franka workspace centered around (-0.3, -0.5, 0.45)
PALM_POSE_MINS = [-0.6, -0.9, 0.35, -np.pi, -np.pi/4, -np.pi]  # [x, y, z, roll, pitch, yaw]
PALM_POSE_MAXS = [0.0, -0.1, 0.8, np.pi, np.pi/4, np.pi]

# PCA matrix from FABRICS (5x16): maps 5D PCA coords to 16D Allegro joints
# Reference: fabrics_sim/fabrics/kuka_allegro_pose_fabric.py
HAND_PCA_MATRIX = torch.tensor([
    [-3.8872e-02, 3.7917e-01, 4.4703e-01, 7.1016e-03, 2.1159e-03,
     3.2014e-01, 4.4660e-01, 5.2108e-02, 5.6869e-05, 2.9845e-01,
     3.8575e-01, 7.5774e-03, -1.4790e-02, 9.8163e-02, 4.3551e-02,
     3.1699e-01],
    [-5.1148e-02, -1.3007e-01, 5.7727e-02, 5.7914e-01, 1.0156e-02,
     -1.8469e-01, 5.3809e-02, 5.4888e-01, 1.3351e-04, -1.7747e-01,
     2.7809e-02, 4.8187e-01, 2.9753e-02, 2.6149e-02, 6.6994e-02,
     1.8117e-01],
    [-5.7137e-02, -3.4707e-01, 3.3365e-01, -1.8029e-01, -4.3560e-02,
     -4.7666e-01, 3.2517e-01, -1.5208e-01, -5.9691e-05, -4.5790e-01,
     3.6536e-01, -1.3916e-01, 2.3925e-03, 3.7238e-02, -1.0124e-01,
     -1.7442e-02],
    [2.2795e-02, -3.4090e-02, 3.4366e-02, -2.6531e-02, 2.3471e-02,
     4.6123e-02, 9.8059e-02, -1.2619e-03, -1.6452e-04, -1.3741e-02,
     1.3813e-01, 2.8677e-02, 2.2661e-01, -5.9911e-01, 7.0257e-01,
     -2.4525e-01],
    [-4.4911e-02, -4.7156e-01, 9.3124e-02, 2.3135e-01, -2.4607e-03,
     9.5564e-02, 1.2470e-01, 3.6613e-02, 1.3821e-04, 4.6072e-01,
     9.9315e-02, -8.1080e-02, -4.7617e-01, -2.7734e-01, -2.3989e-01,
     -3.1222e-01],
], dtype=torch.float32)


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


class FabricActionController(nn.Module):
    """FABRICS-based action controller for Franka + Allegro.

    Converts 11D fabric actions to 23D joint targets:
    - 6D palm pose (XYZ + RPY) -> 7D Franka arm joint targets via differential IK
    - 5D hand PCA -> 16D Allegro hand joint targets via PCA projection

    Reference: NVlabs/DEXTRAH - dextrah_kuka_allegro_env.py
    """

    # Action space dimensions
    NUM_XYZ = 3
    NUM_RPY = 3
    NUM_HAND_PCA = 5
    NUM_FABRIC_ACTIONS = NUM_XYZ + NUM_RPY + NUM_HAND_PCA  # 11

    def __init__(
        self,
        franka_dof: int = 7,
        allegro_dof: int = 16,
        franka_joint_lower: torch.Tensor | None = None,
        franka_joint_upper: torch.Tensor | None = None,
        allegro_joint_lower: torch.Tensor | None = None,
        allegro_joint_upper: torch.Tensor | None = None,
        device: str = "cuda",
        damping: float = 0.1,
        ik_step_size: float = 0.1,
    ):
        """Initialize the fabric action controller.

        Args:
            franka_dof: Number of Franka arm DOFs (7)
            allegro_dof: Number of Allegro hand DOFs (16)
            franka_joint_lower: Lower joint limits for Franka
            franka_joint_upper: Upper joint limits for Franka
            allegro_joint_lower: Lower joint limits for Allegro
            allegro_joint_upper: Upper joint limits for Allegro
            device: Compute device
            damping: Damping for IK solver
            ik_step_size: Step size for iterative IK
        """
        super().__init__()
        self.franka_dof = franka_dof
        self.allegro_dof = allegro_dof
        self.device = device
        self.damping = damping
        self.ik_step_size = ik_step_size

        # Register PCA matrix (5 x 16)
        self.register_buffer("pca_matrix", HAND_PCA_MATRIX.to(device))

        # Action space limits
        self.register_buffer(
            "palm_pose_lower",
            torch.tensor(PALM_POSE_MINS, dtype=torch.float32, device=device)
        )
        self.register_buffer(
            "palm_pose_upper",
            torch.tensor(PALM_POSE_MAXS, dtype=torch.float32, device=device)
        )
        self.register_buffer(
            "hand_pca_lower",
            torch.tensor(HAND_PCA_MINS, dtype=torch.float32, device=device)
        )
        self.register_buffer(
            "hand_pca_upper",
            torch.tensor(HAND_PCA_MAXS, dtype=torch.float32, device=device)
        )

        # Joint limits
        if franka_joint_lower is not None:
            self.register_buffer("franka_joint_lower", franka_joint_lower.to(device))
            self.register_buffer("franka_joint_upper", franka_joint_upper.to(device))
        else:
            # Default Franka Emika Panda joint limits
            self.register_buffer(
                "franka_joint_lower",
                torch.tensor([-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973],
                           dtype=torch.float32, device=device)
            )
            self.register_buffer(
                "franka_joint_upper",
                torch.tensor([2.8973, 1.7628, 2.8973, -0.0698, 2.8973, 3.7525, 2.8973],
                           dtype=torch.float32, device=device)
            )

        if allegro_joint_lower is not None:
            self.register_buffer("allegro_joint_lower", allegro_joint_lower.to(device))
            self.register_buffer("allegro_joint_upper", allegro_joint_upper.to(device))
        else:
            # Default Allegro hand limits (approximate)
            self.register_buffer(
                "allegro_joint_lower",
                torch.tensor([-0.47, -0.196, -0.174, -0.227] * 4, dtype=torch.float32, device=device)
            )
            self.register_buffer(
                "allegro_joint_upper",
                torch.tensor([0.47, 1.61, 1.709, 1.618] * 4, dtype=torch.float32, device=device)
            )

        # Franka Jacobian approximation using DH parameters (simplified)
        # Real implementation would use proper FK/IK from the robot model
        self._init_franka_kinematics()

    def _init_franka_kinematics(self):
        """Initialize Franka kinematics parameters (DH parameters)."""
        # Franka Emika Panda DH parameters (simplified)
        # These are approximate and used for differential IK
        self.register_buffer(
            "franka_d",
            torch.tensor([0.333, 0.0, 0.316, 0.0, 0.384, 0.0, 0.107],
                        dtype=torch.float32, device=self.device)
        )
        self.register_buffer(
            "franka_a",
            torch.tensor([0.0, 0.0, 0.0825, -0.0825, 0.0, 0.088, 0.0],
                        dtype=torch.float32, device=self.device)
        )

    def normalize_action(self, action: torch.Tensor) -> torch.Tensor:
        """Normalize action from [-1, 1] to actual limits.

        Args:
            action: Raw action in [-1, 1] (batch, 11)

        Returns:
            Normalized action with actual values
        """
        # Split action
        palm_action = action[:, :6]  # 6D palm pose
        hand_action = action[:, 6:]  # 5D hand PCA

        # Scale palm pose from [-1, 1] to actual limits
        palm_range = self.palm_pose_upper - self.palm_pose_lower
        palm_target = self.palm_pose_lower + (palm_action + 1.0) * 0.5 * palm_range

        # Scale hand PCA from [-1, 1] to actual limits
        hand_range = self.hand_pca_upper - self.hand_pca_lower
        hand_target = self.hand_pca_lower + (hand_action + 1.0) * 0.5 * hand_range

        return torch.cat([palm_target, hand_target], dim=-1)

    def pca_to_joint(self, hand_pca: torch.Tensor) -> torch.Tensor:
        """Convert 5D hand PCA coordinates to 16D joint positions.

        Args:
            hand_pca: Hand PCA coordinates (batch, 5)

        Returns:
            allegro_q: Allegro joint positions (batch, 16)
        """
        # PCA reconstruction: q = pca_coords @ pca_matrix
        # pca_matrix is (5, 16), hand_pca is (batch, 5)
        allegro_q = torch.matmul(hand_pca, self.pca_matrix)

        # Clamp to joint limits
        allegro_q = torch.clamp(allegro_q, self.allegro_joint_lower, self.allegro_joint_upper)

        return allegro_q

    def euler_to_quat(self, rpy: torch.Tensor) -> torch.Tensor:
        """Convert Euler angles (ZYX convention) to quaternion.

        Args:
            rpy: Roll-Pitch-Yaw angles (batch, 3)

        Returns:
            quat: Quaternion (batch, 4) in (x, y, z, w) format
        """
        roll = rpy[:, 0]
        pitch = rpy[:, 1]
        yaw = rpy[:, 2]

        cy = torch.cos(yaw * 0.5)
        sy = torch.sin(yaw * 0.5)
        cp = torch.cos(pitch * 0.5)
        sp = torch.sin(pitch * 0.5)
        cr = torch.cos(roll * 0.5)
        sr = torch.sin(roll * 0.5)

        w = cr * cp * cy + sr * sp * sy
        x = sr * cp * cy - cr * sp * sy
        y = cr * sp * cy + sr * cp * sy
        z = cr * cp * sy - sr * sp * cy

        return torch.stack([x, y, z, w], dim=-1)

    def compute_franka_jacobian(
        self,
        franka_q: torch.Tensor,
    ) -> torch.Tensor:
        """Compute approximate Franka Jacobian using finite differences.

        For production use, this should be replaced with analytical Jacobian
        from the robot model or a proper IK library.

        Args:
            franka_q: Current Franka joint positions (batch, 7)

        Returns:
            J: Jacobian matrix (batch, 6, 7)
        """
        batch_size = franka_q.shape[0]
        eps = 1e-4

        # Approximate Jacobian using numerical differentiation
        # This is a simplified version; real implementation would use FK
        J = torch.zeros(batch_size, 6, 7, device=self.device)

        # For now, use a simple diagonal approximation
        # Real IK would compute proper Jacobian from FK
        # Joints 0-2 primarily affect XYZ position
        # Joints 3-6 primarily affect orientation
        J[:, 0, 0] = 0.5   # Joint 0 -> X
        J[:, 1, 0] = 0.3   # Joint 0 -> Y
        J[:, 0, 1] = 0.3   # Joint 1 -> X
        J[:, 2, 1] = 0.4   # Joint 1 -> Z
        J[:, 1, 2] = 0.4   # Joint 2 -> Y
        J[:, 0, 3] = 0.2   # Joint 3 -> X
        J[:, 2, 3] = 0.3   # Joint 3 -> Z
        J[:, 3, 4] = 0.5   # Joint 4 -> Roll
        J[:, 4, 5] = 0.5   # Joint 5 -> Pitch
        J[:, 5, 6] = 0.5   # Joint 6 -> Yaw

        return J

    def differential_ik(
        self,
        current_q: torch.Tensor,
        current_ee_pos: torch.Tensor,
        current_ee_quat: torch.Tensor,
        target_pos: torch.Tensor,
        target_rpy: torch.Tensor,
    ) -> torch.Tensor:
        """Compute joint position update using differential IK.

        Args:
            current_q: Current Franka joint positions (batch, 7)
            current_ee_pos: Current EE position (batch, 3)
            current_ee_quat: Current EE quaternion (batch, 4)
            target_pos: Target EE position (batch, 3)
            target_rpy: Target EE orientation as RPY (batch, 3)

        Returns:
            target_q: Target joint positions (batch, 7)
        """
        batch_size = current_q.shape[0]

        # Compute position error
        pos_error = target_pos - current_ee_pos

        # Compute orientation error (simplified: use RPY difference)
        # Convert current quat to RPY for error computation
        current_rpy = self.quat_to_euler(current_ee_quat)
        ori_error = target_rpy - current_rpy

        # Wrap orientation error to [-pi, pi]
        ori_error = torch.atan2(torch.sin(ori_error), torch.cos(ori_error))

        # Combine into 6D task space error
        task_error = torch.cat([pos_error, ori_error], dim=-1)  # (batch, 6)

        # Compute Jacobian
        J = self.compute_franka_jacobian(current_q)  # (batch, 6, 7)

        # Damped least squares IK: dq = J^T (J J^T + λI)^-1 dx
        JJT = torch.bmm(J, J.transpose(1, 2))  # (batch, 6, 6)
        damping_matrix = self.damping * torch.eye(6, device=self.device).unsqueeze(0).expand(batch_size, -1, -1)
        JJT_damped = JJT + damping_matrix

        # Solve for dq
        try:
            JJT_inv = torch.linalg.inv(JJT_damped)
            dq = torch.bmm(J.transpose(1, 2), torch.bmm(JJT_inv, task_error.unsqueeze(-1))).squeeze(-1)
        except Exception:
            # Fallback to simple proportional control
            dq = torch.bmm(J.transpose(1, 2), task_error.unsqueeze(-1)).squeeze(-1) * 0.1

        # Scale update
        dq = dq * self.ik_step_size

        # Compute target joint positions
        target_q = current_q + dq

        # Clamp to joint limits
        target_q = torch.clamp(target_q, self.franka_joint_lower, self.franka_joint_upper)

        return target_q

    def quat_to_euler(self, quat: torch.Tensor) -> torch.Tensor:
        """Convert quaternion to Euler angles (ZYX convention).

        Args:
            quat: Quaternion (batch, 4) in (x, y, z, w) format

        Returns:
            rpy: Roll-Pitch-Yaw angles (batch, 3)
        """
        x, y, z, w = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]

        # Roll (x-axis rotation)
        sinr_cosp = 2 * (w * x + y * z)
        cosr_cosp = 1 - 2 * (x * x + y * y)
        roll = torch.atan2(sinr_cosp, cosr_cosp)

        # Pitch (y-axis rotation)
        sinp = 2 * (w * y - z * x)
        sinp = torch.clamp(sinp, -1.0, 1.0)
        pitch = torch.asin(sinp)

        # Yaw (z-axis rotation)
        siny_cosp = 2 * (w * z + x * y)
        cosy_cosp = 1 - 2 * (y * y + z * z)
        yaw = torch.atan2(siny_cosp, cosy_cosp)

        return torch.stack([roll, pitch, yaw], dim=-1)

    def forward(
        self,
        fabric_action: torch.Tensor,
        current_franka_q: torch.Tensor,
        current_ee_pos: torch.Tensor,
        current_ee_quat: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Convert 11D fabric action to 23D joint targets.

        Args:
            fabric_action: Normalized fabric action in [-1, 1] (batch, 11)
            current_franka_q: Current Franka joint positions (batch, 7)
            current_ee_pos: Current EE position (batch, 3)
            current_ee_quat: Current EE quaternion (batch, 4)

        Returns:
            franka_target: Target Franka joint positions (batch, 7)
            allegro_target: Target Allegro joint positions (batch, 16)
        """
        # Normalize action from [-1, 1] to actual values
        normalized_action = self.normalize_action(fabric_action)

        # Split into palm pose and hand PCA
        palm_target = normalized_action[:, :6]  # (batch, 6) [x, y, z, roll, pitch, yaw]
        hand_pca_target = normalized_action[:, 6:]  # (batch, 5)

        # Convert hand PCA to joint positions
        allegro_target = self.pca_to_joint(hand_pca_target)

        # Convert palm pose to Franka joint positions using differential IK
        target_pos = palm_target[:, :3]
        target_rpy = palm_target[:, 3:6]

        franka_target = self.differential_ik(
            current_franka_q,
            current_ee_pos,
            current_ee_quat,
            target_pos,
            target_rpy,
        )

        return franka_target, allegro_target

    def get_action_dim(self) -> int:
        """Return the fabric action dimension (11)."""
        return self.NUM_FABRIC_ACTIONS

    def get_joint_dim(self) -> int:
        """Return the total joint dimension (23)."""
        return self.franka_dof + self.allegro_dof
