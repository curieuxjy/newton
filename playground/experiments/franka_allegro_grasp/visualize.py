"""Visualization script for Franka + Allegro cube grasping environment.

Run with:
    uv run --extra examples --extra torch-cu12 python -m playground.experiments.franka_allegro_grasp.visualize

With depth visualization (requires matplotlib):
    uv run --extra examples --extra torch-cu12 --with matplotlib python -m playground.experiments.franka_allegro_grasp.visualize --use-depth --show-depth
"""

import argparse
import math

import numpy as np
import torch
import warp as wp

import newton
import newton.examples

from .config import EnvConfig
from .env import FrankaAllegroGraspEnv


class VisualizeExample:
    """Visualization wrapper for the grasping environment."""

    def __init__(self, viewer, args):
        self.viewer = viewer
        self.args = args
        self.config = EnvConfig(
            num_envs=args.num_envs,
            use_depth_sensor=args.use_depth,
        )

        # Create environment
        self.env = FrankaAllegroGraspEnv(self.config, device="cuda", headless=False)

        # Set viewer model
        self.viewer.set_model(self.env.model)

        # Disable visual world offsets (we use physical spacing)
        self.viewer.set_world_offsets((0.0, 0.0, 0.0))

        # Reset environment
        self.env.reset()

        # Timing
        self.fps = self.config.fps
        self.frame_dt = 1.0 / self.fps
        self.sim_time = 0.0

        # Load checkpoint if provided
        self.policy = None
        if args.checkpoint:
            self._load_checkpoint(args.checkpoint)

        # Setup matplotlib depth visualization if requested
        self.depth_fig = None
        self.depth_ims = None
        if args.show_depth and args.use_depth:
            self._setup_depth_visualization()

        # For scripted demo
        self.demo_phase = 0
        self.demo_time = 0.0

    def _setup_depth_visualization(self):
        """Setup matplotlib figure for depth visualization."""
        try:
            import matplotlib.pyplot as plt

            # Create figure with 2x2 grid
            self.depth_fig, axes = plt.subplots(2, 2, figsize=(8, 8))
            self.depth_fig.suptitle("Depth Images (Real-time)", fontsize=12)

            self.depth_ims = []
            for i in range(4):
                ax = axes[i // 2, i % 2]
                im = ax.imshow(
                    np.zeros((self.config.depth_height, self.config.depth_width)),
                    cmap='viridis',
                    vmin=self.config.depth_min,
                    vmax=2.0
                )
                ax.set_title(f"Env {i}")
                ax.axis('off')
                plt.colorbar(im, ax=ax, fraction=0.046)
                self.depth_ims.append(im)

            plt.tight_layout()
            plt.ion()
            plt.show(block=False)
            print("[INFO] Depth visualization window opened")

        except ImportError:
            print("[WARNING] matplotlib not installed. Run with: --with matplotlib")
            self.depth_fig = None
            self.depth_ims = None

    def _update_depth_visualization(self):
        """Update matplotlib depth display."""
        if self.depth_fig is None or self.depth_ims is None:
            return

        import matplotlib.pyplot as plt

        # Check if figure still exists
        if not plt.fignum_exists(self.depth_fig.number):
            self.depth_fig = None
            self.depth_ims = None
            return

        # Get depth data
        depth_data = self.env.depth_image.numpy()
        depth_reshaped = depth_data.reshape(
            self.env.num_envs, 1, self.config.depth_height, self.config.depth_width
        )

        # Update images
        for i in range(min(4, self.env.num_envs)):
            depth = depth_reshaped[i, 0]
            # Flip vertically to correct orientation
            depth = np.flipud(depth)
            depth_clipped = np.clip(depth, 0, 2.0)
            self.depth_ims[i].set_data(depth_clipped)

        self.depth_fig.canvas.draw_idle()
        self.depth_fig.canvas.flush_events()

    def _load_checkpoint(self, path: str):
        """Load trained policy from checkpoint."""
        try:
            import torch.nn as nn

            # Simple MLP policy (matches PPO architecture)
            class Policy(nn.Module):
                def __init__(self, obs_dim, act_dim, hidden_dims=(512, 256, 128)):
                    super().__init__()
                    layers = []
                    in_dim = obs_dim
                    for h_dim in hidden_dims:
                        layers.append(nn.Linear(in_dim, h_dim))
                        layers.append(nn.ELU())
                        in_dim = h_dim
                    layers.append(nn.Linear(in_dim, act_dim))
                    layers.append(nn.Tanh())
                    self.net = nn.Sequential(*layers)

                def forward(self, x):
                    return self.net(x)

            checkpoint = torch.load(path, map_location="cuda")
            self.policy = Policy(self.env.num_obs, self.env.num_actions)
            self.policy.load_state_dict(checkpoint["policy"])
            self.policy.eval()
            self.policy.cuda()
            print(f"[INFO] Loaded policy from {path}")
        except Exception as e:
            print(f"[WARNING] Failed to load checkpoint: {e}")
            self.policy = None

    def _scripted_action(self) -> torch.Tensor:
        """Generate scripted demo action for visualization."""
        actions = torch.zeros(self.env.num_envs, self.env.num_actions, device="cuda")

        # Simple reaching motion
        t = self.demo_time

        # Phase 0: Move arm forward (first 3 seconds)
        if t < 3.0:
            # Franka: extend arm forward
            actions[:, 0] = 0.0   # Joint 1
            actions[:, 1] = 0.2   # Joint 2 - lower
            actions[:, 2] = 0.0   # Joint 3
            actions[:, 3] = -0.3  # Joint 4 - extend
            actions[:, 4] = 0.0   # Joint 5
            actions[:, 5] = 0.2   # Joint 6
            actions[:, 6] = 0.0   # Joint 7

        # Phase 1: Close fingers (3-6 seconds)
        elif t < 6.0:
            # Allegro: close fingers
            finger_close = min((t - 3.0) / 3.0, 1.0) * 0.5
            for i in range(7, 23):
                if (i - 7) % 4 != 0:  # Skip abduction joints
                    actions[:, i] = finger_close

        # Phase 2: Lift (6+ seconds)
        else:
            # Franka: lift up
            actions[:, 1] = -0.3  # Joint 2 - raise
            actions[:, 3] = 0.2   # Joint 4 - bend up

            # Keep fingers closed
            for i in range(7, 23):
                if (i - 7) % 4 != 0:
                    actions[:, i] = 0.3

        return actions

    def step(self):
        """Step the environment."""
        self.demo_time += self.frame_dt

        # Get action
        if self.policy is not None:
            with torch.no_grad():
                obs = self.env._compute_observations()
                actions = self.policy(obs)
        elif self.args.random:
            actions = torch.randn(self.env.num_envs, self.env.num_actions, device="cuda") * 0.1
        else:
            actions = self._scripted_action()

        # Step environment
        obs, rewards, dones, info = self.env.step(actions)

        self.sim_time += self.frame_dt

        # Log info periodically
        if int(self.sim_time * 10) % 10 == 0:
            rc = self.env.reward_components
            print(f"[t={self.sim_time:.1f}s] "
                  f"phase: R={rc.get('phase_reach', 0):.1f}/G={rc.get('phase_grasp', 0):.1f}/L={rc.get('phase_lift', 0):.1f} | "
                  f"ee2cube: {rc.get('ee_to_cube_dist', 0):.3f} | "
                  f"cube_h: {rc.get('cube_height', 0):.3f}")

    def render(self):
        """Render the scene."""
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.env.state_0)
        self.viewer.log_contacts(self.env.contacts, self.env.state_0)

        # Draw goal markers (skip in headless mode for performance)
        if not self.args.headless:
            self._draw_goal_markers()

        # Draw camera debug visualization
        if self.args.debug_camera and self.args.use_depth:
            self._draw_camera_debug()

        self.viewer.end_frame()

        # Update depth visualization if enabled
        if self.args.show_depth and self.args.use_depth:
            self._update_depth_visualization()

    def _draw_camera_debug(self):
        """Draw camera position and field of view for debugging."""
        grid_size = int(np.ceil(np.sqrt(self.env.num_envs)))

        # Camera visualization parameters
        cam_marker_size = 0.05  # Size of camera position marker
        fov_length = 0.5   # Length of FOV lines
        fov_angle = math.radians(self.config.depth_fov / 2)

        line_starts = []
        line_ends = []
        line_colors = []

        for i in range(min(self.env.num_envs, 4)):
            # Calculate environment offset
            grid_x = i // grid_size
            grid_y = i % grid_size
            env_offset_x = grid_x * self.env.env_spacing[0]
            env_offset_y = grid_y * self.env.env_spacing[1]

            # Camera position (same as in env.py _update_depth_sensor)
            # Opposite short side of table (near robot side), lower height
            cam_x = -0.3 + env_offset_x   # Centered on table X
            cam_y = 0.1 + env_offset_y    # Beyond the near Y edge of table
            cam_z = 0.43                   # Lower height (closer to table level)
            cam_pos = np.array([cam_x, cam_y, cam_z], dtype=np.float32)

            # Target position (cube area)
            target_x = -0.3 + env_offset_x
            target_y = -0.5 + env_offset_y
            target_z = 0.45  # Cube height
            target_pos = np.array([target_x, target_y, target_z], dtype=np.float32)

            # Draw camera position marker as 3D cross (orange)
            for axis, color in [
                (np.array([1, 0, 0], dtype=np.float32), (1.0, 0.5, 0.0)),
                (np.array([0, 1, 0], dtype=np.float32), (1.0, 0.5, 0.0)),
                (np.array([0, 0, 1], dtype=np.float32), (1.0, 0.5, 0.0)),
            ]:
                line_starts.append(cam_pos - axis * cam_marker_size)
                line_ends.append(cam_pos + axis * cam_marker_size)
                line_colors.append(color)

            # Calculate camera direction
            forward = target_pos - cam_pos
            forward = forward / np.linalg.norm(forward)

            # Calculate up and right vectors
            world_up = np.array([0.0, 0.0, 1.0], dtype=np.float32)
            right = np.cross(forward, world_up)
            right = right / (np.linalg.norm(right) + 1e-6)
            up = np.cross(right, forward)
            up = up / (np.linalg.norm(up) + 1e-6)

            # Draw center line (camera to target) - Yellow
            line_starts.append(cam_pos)
            line_ends.append(cam_pos + forward * fov_length)
            line_colors.append((1.0, 1.0, 0.0))

            # Draw FOV frustum lines (4 corners) - Cyan
            tan_fov = math.tan(fov_angle)
            corners = [
                forward + (up + right) * tan_fov,
                forward + (up - right) * tan_fov,
                forward + (-up + right) * tan_fov,
                forward + (-up - right) * tan_fov,
            ]

            corner_points = []
            for corner in corners:
                corner_normalized = corner / np.linalg.norm(corner)
                corner_points.append(cam_pos + corner_normalized * fov_length)
                line_starts.append(cam_pos)
                line_ends.append(cam_pos + corner_normalized * fov_length)
                line_colors.append((0.0, 1.0, 1.0))

            # Connect corners to form rectangle at FOV end
            for j in range(4):
                line_starts.append(corner_points[j])
                line_ends.append(corner_points[(j + 1) % 4])
                line_colors.append((0.0, 1.0, 1.0))

        # Log all lines
        if line_starts:
            starts_wp = wp.array(np.array(line_starts, dtype=np.float32), dtype=wp.vec3f)
            ends_wp = wp.array(np.array(line_ends, dtype=np.float32), dtype=wp.vec3f)
            colors_wp = wp.array(np.array(line_colors, dtype=np.float32), dtype=wp.vec3f)
            self.viewer.log_lines("/camera_debug", starts_wp, ends_wp, colors_wp)

    def _draw_goal_markers(self):
        """Draw goal position markers."""
        goal_pos = self.env.goal_pos.cpu().numpy()

        starts = []
        ends = []
        colors = []

        axis_length = 0.1
        for i in range(min(self.env.num_envs, 16)):  # Limit for performance
            pos = goal_pos[i]

            # X axis (red)
            starts.append(pos)
            ends.append(pos + np.array([axis_length, 0, 0]))
            colors.append((1.0, 0.0, 0.0))

            # Y axis (green)
            starts.append(pos)
            ends.append(pos + np.array([0, axis_length, 0]))
            colors.append((0.0, 1.0, 0.0))

            # Z axis (blue)
            starts.append(pos)
            ends.append(pos + np.array([0, 0, axis_length]))
            colors.append((0.0, 0.0, 1.0))

        if starts:
            starts_wp = wp.array(np.array(starts, dtype=np.float32), dtype=wp.vec3f)
            ends_wp = wp.array(np.array(ends, dtype=np.float32), dtype=wp.vec3f)
            colors_wp = wp.array(np.array(colors, dtype=np.float32), dtype=wp.vec3f)
            self.viewer.log_lines("/goal_markers", starts_wp, ends_wp, colors_wp)

    def test_final(self):
        """Validation test."""
        pass


def main():
    # Use newton's parser as base
    parser = newton.examples.create_parser()
    parser.add_argument("--num-envs", type=int, default=4, help="Number of environments")
    parser.add_argument("--no-depth", dest="use_depth", action="store_false", default=True, help="Disable depth sensor")
    parser.add_argument("--no-show-depth", dest="show_depth", action="store_false", default=True, help="Hide real-time depth visualization")
    parser.add_argument("--no-debug-camera", dest="debug_camera", action="store_false", default=True, help="Hide camera position and FOV debug visualization")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to trained policy checkpoint")
    parser.add_argument("--random", action="store_true", help="Use random actions")
    parser.add_argument("--show-collision", action="store_true", help="Show collision shapes")

    # Initialize viewer with merged parser
    viewer, args = newton.examples.init(parser)

    # Create and run example
    example = VisualizeExample(viewer, args)
    newton.examples.run(example, args)


if __name__ == "__main__":
    main()
