# Franka Arm + Allegro Hand Combined Robot

Franka Emika Panda 로봇 팔에 Allegro 손을 결합한 manipulation 시스템 예제입니다.

## 개요

이 예제는 두 개의 독립적인 URDF 모델(Franka arm, Allegro hand)을 하나의 kinematic chain으로 결합하는 방법을 보여줍니다.

## 로봇 구조

### Franka Emika Panda (FR3)

- **DOF**: 7 (revolute joints)
- **Base**: 원점에 고정
- **End Effector**: `fr3_link8`

### Allegro Hand (Left)

- **DOF**: 16 (4 fingers × 4 joints)
- **Attachment**: Franka EE에서 Z축으로 0.1m 오프셋, 180° 회전

## DOF 매핑

Newton은 URDF를 kinematic tree 순서로 로드합니다. Allegro hand의 DOF 매핑:

| DOF | 손가락 | URDF Links | 관절 구조 |
|-----|--------|------------|-----------|
| 0-3 | Ring (약지) | link_8-11 | Abduction, MCP, PIP, DIP |
| 4-7 | Middle (중지) | link_4-7 | Abduction, MCP, PIP, DIP |
| 8-11 | Index (검지) | link_0-3 | Abduction, MCP, PIP, DIP |
| 12-15 | Thumb (엄지) | link_12-15 | Pitch, Roll, MCP, PIP |

### 관절 타입 설명

- **Abduction**: 손바닥에 가장 가까운 관절, 손가락을 좌우로 벌리는 동작
- **MCP** (Metacarpophalangeal): 손가락 굽힘의 첫 번째 관절
- **PIP** (Proximal Interphalangeal): 손가락 굽힘의 두 번째 관절
- **DIP** (Distal Interphalangeal): 손가락 끝 관절

### 관절 한계 (radians)

| 관절 타입 | Lower | Upper | Range |
|-----------|-------|-------|-------|
| Abduction | -0.470 | 0.470 | 0.940 |
| MCP | -0.196 | 1.610 | 1.806 |
| PIP | -0.174 | 1.709 | 1.883 |
| DIP | -0.227 | 1.618 | 1.845 |

## 실행 방법

```bash
# 기본 실행 (GUI)
uv run --extra examples python -m playground.experiments.franka_allegro.example_franka_allegro

# Headless 모드
uv run --extra examples python -m playground.experiments.franka_allegro.example_franka_allegro --headless
```

## 구현 세부사항

### Solver 설정

`SolverMuJoCo`를 사용하며, 다관절 로봇에 적합한 파라미터를 설정합니다:

```python
self.solver = newton.solvers.SolverMuJoCo(
    self.model,
    use_mujoco_contacts=False,
    solver="newton",
    integrator="implicitfast",
    cone="elliptic",
    njmax=15000,
    nconmax=15000,
    iterations=15,
    ls_iterations=100,
    ls_parallel=True,
    impratio=1000.0,
)
```

### 관절 파라미터

#### Franka Arm

```python
joint_target_ke = 500.0  # Stiffness
joint_target_kd = 100.0  # Damping
joint_effort_limit = 80.0
joint_armature = 0.1
```

#### Allegro Hand

```python
joint_target_ke = 200.0  # Higher stiffness for stability
joint_target_kd = 20.0
joint_effort_limit = 20.0
joint_armature = 0.05
```

### Articulation 병합

두 URDF를 하나의 articulation으로 결합하는 핵심 코드:

```python
# 1. Allegro root joint의 parent를 Franka EE로 설정
allegro_root_joint_idx = joint_offset
builder.joint_parent[allegro_root_joint_idx] = ee_body_idx
builder.joint_X_p[allegro_root_joint_idx] = allegro_xform

# 2. 모든 Allegro joints를 같은 articulation에 할당
for j in range(joint_offset, builder.joint_count):
    builder.joint_articulation[j] = 0

# 3. Articulation 메타데이터 업데이트
builder.articulation_start = [0]
builder.articulation_key = ["franka_allegro"]
```

### CUDA Graph 비활성화

제어 타겟이 매 프레임 변경되므로 CUDA graph capture를 비활성화합니다:

```python
def capture(self):
    # Control targets change every frame, disable CUDA graph
    self.graph = None
```

## 애니메이션

현재 예제는 Ring, Middle, Index 세 손가락을 동시에 sine wave로 움직여 동작을 비교할 수 있습니다:

```python
for finger_idx in range(3):  # Ring, Middle, Index
    start_dof = finger_idx * 4
    for j in range(4):
        dof = start_dof + j
        t = np.sin(self.anim_time * 3.0)
        allegro_target[dof] = allegro_mid[dof] + t * allegro_range[dof] * 0.9
```

## 참고 자료

- [Franka Emika Panda](https://www.franka.de/)
- [Wonik Allegro Hand](https://www.wonikrobotics.com/robot-hand-allegro-hand)
- Newton 예제: `newton/examples/robot/example_robot_panda_hydro.py`
