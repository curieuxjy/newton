# xArm7 + Allegro Cube Grasping

UFactory xArm7 arm + Wonik Allegro hand grasping a cube on a table.
A drop-in variant of [`franka_allegro_grasp`](../franka_allegro_grasp/) with
the Franka FR3 swapped for an xArm7 (mujoco_menagerie). All RL / FABRICS /
distillation / network code is reused from the franka package — only the
arm asset, kinematics, and a few task-specific defaults are overridden.

## Quickstart

```bash
# Visualize (4 envs, scripted demo actions)
uv run --extra examples --extra torch-cu12 \
    python -m playground.experiments.xarm7_allegro_grasp.visualize

# Random actions (each env diverges — useful to verify parallelism)
uv run --extra examples --extra torch-cu12 \
    python -m playground.experiments.xarm7_allegro_grasp.visualize --random

# Teacher PPO training
uv run --extra examples --extra torch-cu12 \
    python -m playground.experiments.xarm7_allegro_grasp.train

# Save / view depth-camera output
uv run --extra examples --extra torch-cu12 \
    python -m playground.experiments.xarm7_allegro_grasp.view_depth
```

The first run downloads `mujoco_menagerie` (~150 MB, blob-filtered) into
`~/.cache/newton/mujoco_menagerie_<sha>/`. To reuse an existing clone:

```bash
export NEWTON_MENAGERIE_PATH=/path/to/your/mujoco_menagerie
```

## Environment

| | Value |
|---|---|
| Arm | UFactory xArm7 (7 DOF, mujoco_menagerie `xarm7_nohand.xml`) |
| Hand | Wonik Allegro (16 DOF, left, newton-assets) |
| Total robot DOF | 23 (7 arm + 16 hand) |
| Object | 5 cm cube, 0.1 kg, free joint |
| Table | 0.6 × 0.8 × 0.02 m, top at z ≈ 0.21 m |
| Solver | `SolverMuJoCo` (Newton backend, implicitfast) |
| Sensors | tiled depth camera (160×120, 48° FOV), fingertip contact |
| Default num envs | 4 (visualize) / 256 (train) — RTX 5090 limit |

The observation/reward layout (159-D student, 172-D teacher, lift-target
goal, 23-D direct or 11-D FABRICS action space) is identical to
`franka_allegro_grasp`. See its README/docs for details.

## Files

| File | Purpose |
|---|---|
| `__init__.py` | Package exports (re-exports configs and networks from the franka package) |
| `_menagerie.py` | Tiny git-clone helper for `mujoco_menagerie` (pinned ref) |
| `_xarm7_fabric.py` | `XArm7FabricActionController` — xArm7 modified DH for IK |
| `config.py` | `make_xarm7_env_config` / `make_xarm7_train_config` factories |
| `env.py` | `XArm7AllegroGraspEnv(FrankaAllegroGraspEnv)` — overrides `_build_simulation`, `__init__`, `reset` |
| `train.py` | Thin wrapper that monkey-patches the franka training loop with the xArm7 env |
| `visualize.py` | Same wrapper pattern for the visualizer |
| `view_depth.py` | Same wrapper pattern for the depth viewer |

## Design notes

### What was overridden vs reused

The franka package is monolithic (~4800 lines). To avoid duplication, only
arm-specific pieces are overridden:

- **Arm asset**: `add_urdf("fr3.urdf")` → `add_mjcf("xarm7_nohand.xml")`.
  The `_nohand` variant is used so the parent's hardcoded
  `7 (arm) + 16 (hand) = 23` DOF layout still holds.
- **Init pose / EE link search**: `XARM7_INIT_Q` and `XARM7_EE_LINK_KEY`
  class attributes on `XArm7AllegroGraspEnv`.
- **Arm gains**: stiffness/damping/effort/armature defaults in
  `make_xarm7_env_config`. Field names retain the `franka_*` prefix because
  the parent class indexes them by name; in this package they mean *arm*.
- **Cube spawn z**: corrected to rest on the table top.
- **`reset()` goal**: parent's `goal_pos.z = table_height + lift_height`
  formula floats too high relative to where the cube actually rests, so the
  override recomputes the lift target as `cube_z + lift_height`.
- **FABRICS IK**: parent's `FabricActionController` hardcodes Franka FR3
  modified DH constants. `XArm7FabricActionController` re-registers
  `dh_a`/`dh_d`/`dh_alpha` with xArm7 values; the FK loop, geometric
  Jacobian, and damped-least-squares IK are reused unchanged.
- **Entry-point wiring**: `train.py`/`visualize.py`/`view_depth.py`
  monkey-patch `FrankaAllegroGraspEnv` and `EnvConfig` in the franka modules
  before delegating to their `main()`. This keeps argparse, the training
  loop, the W&B logging, and the matplotlib depth viewer 100% reused.

### xArm7 modified DH (Craig convention, m / rad)

```
i  | a_{i-1} | d_i    | alpha_{i-1}
---|---------|--------|------------
1  | 0       | 0.267  | 0
2  | 0       | 0      | -π/2
3  | 0.0525  | 0      |  π/2
4  | 0.0775  | 0.293  |  π/2
5  | 0       | 0      |  π/2
6  | 0.076   | 0      |  π/2
7  | 0       | 0.097  | -π/2
```

Stored as 8-entry tensors (`dh_a`, `dh_d`, `dh_alpha`) — 7 joints + a
trailing zero "flange" step — to match the parent's
`for i in range(8)` FK loop.

### Reused unchanged from `franka_allegro_grasp`

- `fabric.GraspFabric` — reward features (palm/fingertip distances). Takes
  `ee_pos`/`ee_quat` from the simulation state, so it's arm-agnostic.
- `networks.{TeacherActorCritic, StudentNetwork, ObsRunningMeanStd}`
- `train.TeacherTrainer` and the entire PPO loop
- `distill` (DEXTRAH-G two-stage student distillation)
- `visualize.VisualizeExample` (scripted demo, checkpoint loading,
  matplotlib depth view, debug overlays)
- `view_depth.{run_realtime_visualization, run_save_mode}`

## Known limitations / things to verify

1. **Allegro mount transform** (`ALLEGRO_MOUNT_XFORM` in `env.py`) reuses
   the Franka offset (`Z + 10 cm`, 180° about Z). xArm7 link7 has a
   different orientation than FR3 link8, so the hand may need a small
   rotation tweak — eyeball it in `visualize` and adjust.
2. **Camera-debug `target_z`** in `franka_allegro_grasp/visualize.py:338`
   is hardcoded at 0.45 m and only affects the FOV debug lines when
   `--use-depth` is on. Pass `--no-debug-camera` to hide.
3. **xArm7 base frame**: the DH table assumes a standard z-up base. If the
   loaded MJCF rotates the base, IK targets will be skewed.
4. **EE definition**: `dh_d[7] = 0.097` puts the kinematic EE at the
   flange face. Newton's `ee_body_idx` is `link7`, so there can be a
   ~cm-scale offset between the IK target and the simulated EE pose.
   Adjust `dh_d` index 6 if needed.
5. **Joint sign conventions**: if a joint moves the wrong way under
   FABRICS, that joint's sign needs to be flipped before being fed to FK.

## References

- Parent example: `playground/experiments/franka_allegro_grasp/`
- DEXTRAH-G: https://github.com/NVlabs/DEXTRAH
- mujoco_menagerie xArm7: https://github.com/google-deepmind/mujoco_menagerie/tree/main/ufactory_xarm7
- Pinned menagerie ref: `feadf76d42f8a2162426f7d226a3b539556b3bf5`
