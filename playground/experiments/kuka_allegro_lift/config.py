"""Configuration for Kuka iiwa 14 + Allegro Hand cube lift environment.

Replicates the Isaac-Dexsuite-Kuka-Allegro-Lift-v0 task from IsaacLab
using the Newton physics engine.
"""

from dataclasses import dataclass


@dataclass
class EnvConfig:
    """Environment configuration."""

    num_envs: int = 1
    episode_length: int = 600  # 10 seconds at 60Hz

    # Simulation
    fps: int = 60
    sim_substeps: int = 10
    sim_dt_override: float | None = None  # If set, overrides computed dt

    # Robot - Kuka iiwa 14 (high gains needed for MJCF model with Allegro load)
    kuka_stiffness: float = 100000.0
    kuka_damping: float = 10000.0
    kuka_effort_limit: float = 5000.0
    kuka_armature: float = 0.1

    # Robot - Allegro hand
    hand_stiffness: float = 200.0
    hand_damping: float = 20.0
    hand_effort_limit: float = 20.0
    hand_armature: float = 0.05

    # Action
    action_scale: float = 0.1
    use_relative_control: bool = True

    # Table (surface at z ≈ 0.21)
    table_height: float = 0.4
    table_size: tuple = (0.6, 0.6, 0.02)
    table_pos: tuple = (0.0, -0.62)

    # Cube (rests on table at z ≈ 0.235)
    cube_size: float = 0.05
    cube_mass: float = 0.1
    cube_spawn_pos: tuple = (0.0, -0.62, 0.25)  # Above table center
    cube_spawn_noise: float = 0.03

    # Goal
    lift_height: float = 0.15
    goal_tolerance: float = 0.1

    # Reward weights
    hand_to_object_weight: float = 1.0
    hand_to_object_sharpness: float = 10.0
    object_to_goal_weight: float = 5.0
    object_to_goal_sharpness: float = 15.0
    lift_weight: float = 5.0
    lift_sharpness: float = 8.5
    finger_curl_reg_weight: float = -0.01
    reward_scale: float = 0.01

    # Termination
    fall_height: float = 0.2
    workspace_margin: float = 0.15

    # Domain randomization
    randomize_cube_pos: bool = True
