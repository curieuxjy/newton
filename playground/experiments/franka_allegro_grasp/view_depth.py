"""Depth image visualization script.

Run with:
    uv run --extra examples --extra torch-cu12 python -m playground.experiments.franka_allegro_grasp.view_depth

Options:
    --realtime    Show real-time depth visualization with matplotlib
    --save        Save depth images as PNG files
"""

import argparse

import numpy as np
import torch
import warp as wp

from .config import EnvConfig
from .env import FrankaAllegroGraspEnv


def depth_to_colormap(depth: np.ndarray, vmin: float, vmax: float) -> np.ndarray:
    """Convert depth values to a colormap (viridis-like)."""
    # Normalize to 0-1
    normalized = np.clip((depth - vmin) / (vmax - vmin), 0, 1)

    # Simple viridis-like colormap
    r = np.clip(0.267 + 0.004 * normalized + 1.067 * normalized**2, 0, 1)
    g = np.clip(0.004 + 1.217 * normalized - 0.357 * normalized**2, 0, 1)
    b = np.clip(0.329 + 1.071 * normalized - 1.400 * normalized**2, 0, 1)

    # Stack to RGB
    rgb = np.stack([r, g, b], axis=-1)
    return (rgb * 255).astype(np.uint8)


def save_depth_image(depth: np.ndarray, path: str, vmin: float, vmax: float):
    """Save depth image as PNG using PIL."""
    from PIL import Image

    # Convert to colormap
    rgb = depth_to_colormap(depth, vmin, vmax)

    # Create and save image
    img = Image.fromarray(rgb)
    img = img.resize((256, 256), Image.Resampling.NEAREST)  # Upscale for visibility
    img.save(path)


def run_realtime_visualization(env, config):
    """Run real-time depth visualization with matplotlib."""
    import matplotlib.pyplot as plt

    # Setup figure with 2x2 grid for 4 environments
    fig, axes = plt.subplots(2, 2, figsize=(10, 10))
    fig.suptitle("Real-time Depth Images (Close window to exit)", fontsize=14)

    # Initialize images
    ims = []
    for i in range(4):
        ax = axes[i // 2, i % 2]
        im = ax.imshow(
            np.zeros((config.depth_height, config.depth_width)),
            cmap='viridis',
            vmin=config.depth_min,
            vmax=2.0  # Limit max for better visualization
        )
        ax.set_title(f"Env {i}")
        ax.axis('off')
        plt.colorbar(im, ax=ax, label='Depth (m)', fraction=0.046)
        ims.append(im)

    plt.tight_layout()
    plt.ion()  # Interactive mode
    plt.show()

    print("\nRunning real-time visualization...")
    print("Close the matplotlib window to exit.")

    actions = torch.zeros(env.num_envs, env.num_actions, device="cuda")
    step = 0

    try:
        while plt.fignum_exists(fig.number):
            # Step environment with small random actions
            noise = torch.randn_like(actions) * 0.05
            obs, rewards, dones, info = env.step(actions + noise)

            # Get depth data - new API shape is (num_worlds, num_cameras, height, width)
            depth_data = env.depth_image.numpy()
            # Shape is already (num_envs, 1, height, width)
            depth_reshaped = depth_data

            # Update images
            for i in range(min(4, env.num_envs)):
                depth = depth_reshaped[i, 0]
                # Flip vertically to correct orientation
                depth = np.flipud(depth)
                # Clip large values for better visualization
                depth_clipped = np.clip(depth, 0, 2.0)
                ims[i].set_data(depth_clipped)

            fig.canvas.draw_idle()
            fig.canvas.flush_events()

            step += 1
            if step % 30 == 0:
                print(f"Step {step}, mean depth: {depth_reshaped[0, 0].mean():.3f}m")

            plt.pause(0.01)

    except KeyboardInterrupt:
        print("\nStopped by user.")

    plt.close(fig)
    print("Visualization closed.")


def run_save_mode(env, config):
    """Save depth images to files."""
    actions = torch.zeros(env.num_envs, env.num_actions, device="cuda")

    # Run a few steps to stabilize
    for _ in range(5):
        env.step(actions)

    # Get depth image from the sensor
    # New API shape is (num_worlds, num_cameras, height, width)
    depth_data = env.depth_image.numpy()
    print(f"Depth shape: {depth_data.shape}")

    # Shape is already (num_envs, 1, height, width)
    depth_reshaped = depth_data

    # Save depth images
    print("\nSaving depth images...")
    for i in range(min(4, env.num_envs)):
        depth = depth_reshaped[i, 0]
        # Flip vertically to correct orientation
        depth = np.flipud(depth)
        output_path = f"/tmp/depth_env{i}.png"
        save_depth_image(depth, output_path, config.depth_min, config.depth_max)
        print(f"  Env {i}: saved to {output_path}")

    # Print statistics
    print("\nDepth Statistics:")
    for i in range(min(4, env.num_envs)):
        depth = depth_reshaped[i, 0]
        # Flip vertically to correct orientation
        depth = np.flipud(depth)
        valid_depth = depth[depth > 0]
        if len(valid_depth) > 0:
            print(f"  Env {i}: min={valid_depth.min():.3f}m, max={valid_depth.max():.3f}m, mean={valid_depth.mean():.3f}m")
        else:
            print(f"  Env {i}: No valid depth values")

    # Save raw depth as numpy
    np.save("/tmp/depth_raw.npy", depth_data)
    print(f"\nRaw depth data saved to: /tmp/depth_raw.npy")


def main():
    parser = argparse.ArgumentParser(description="Depth image visualization")
    parser.add_argument("--realtime", action="store_true", help="Show real-time matplotlib visualization")
    parser.add_argument("--save", action="store_true", help="Save depth images as PNG")
    args = parser.parse_args()

    # Default to save mode if no option specified
    if not args.realtime and not args.save:
        args.save = True

    wp.init()

    # Create environment with depth sensor
    config = EnvConfig(
        num_envs=4,
        use_depth_sensor=True,
    )
    env = FrankaAllegroGraspEnv(config, device="cuda", headless=True)

    print(f"Depth image size: {config.depth_width}x{config.depth_height}")
    print(f"Depth range: [{config.depth_min}, {config.depth_max}] meters")

    # Reset environment
    env.reset()

    if args.realtime:
        run_realtime_visualization(env, config)
    elif args.save:
        run_save_mode(env, config)


if __name__ == "__main__":
    main()
