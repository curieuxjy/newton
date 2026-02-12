# Newton Simulator Updates Tracking

이 문서는 Newton 물리 시뮬레이터 메인 팀의 업데이트를 추적합니다.
**업데이트 요청 시에만 기록됩니다.**

---

## 2026-02-12 업데이트

### 커밋 범위
`2fcc770..f1e207f` (main branch merge)

### 주요 변경 사항

#### 1. ⚠️ CollisionPipelineUnified 제거 (#1538)
**커밋**: `27a9bde`

**중요**: `CollisionPipelineUnified` 클래스가 완전히 제거됨. `CollisionPipeline`으로 통합.

```python
# Old
from newton import CollisionPipelineUnified
pipeline = CollisionPipelineUnified.from_model(model)

# New
from newton import CollisionPipeline, BroadPhaseMode
pipeline = CollisionPipeline.from_model(model, broad_phase_mode=BroadPhaseMode.SAP)
# 또는 BroadPhaseMode.NXN, BroadPhaseMode.EXPLICIT
```

- `newton/_src/sim/collide_unified.py` 파일 삭제 (778줄)
- `newton/__init__.py`에서 export 제거
- 모든 예제 및 테스트 업데이트됨

**영향**: `CollisionPipelineUnified`를 사용하는 모든 코드 마이그레이션 필요.

#### 2. ⚠️ 파라미터 이름 변경: `I_m` → `inertia` (#1551)
**커밋**: `1d9e852`

| 메서드 | Old | New |
|--------|-----|-----|
| `ModelBuilder.add_link()` | `I_m=...` | `inertia=...` |
| `ModelBuilder.add_body()` | `I_m=...` | `inertia=...` |
| `SDF.__init__()` | `I=...` | `inertia=...` |

```python
# Old
body = builder.add_link(mass=1.0, I_m=wp.mat33(np.eye(3)))

# New
body = builder.add_link(mass=1.0, inertia=wp.mat33(np.eye(3)))
```

**영향**: `I_m` 또는 `I` 파라미터를 사용하는 코드 수정 필요.

#### 3. ⚠️ `MESH_MAXHULLVERT` → `Mesh.MAX_HULL_VERTICES` (#1598)
**커밋**: `7c9cdfb`

```python
# Old
from newton.geometry import MESH_MAXHULLVERT
max_verts = MESH_MAXHULLVERT  # 64

# New
from newton import Mesh
max_verts = Mesh.MAX_HULL_VERTICES  # 64
```

- 모듈 레벨 상수에서 클래스 속성으로 이동
- `newton._src/geometry/__init__.py`에서 export 제거

**영향**: `MESH_MAXHULLVERT`를 import하는 코드 수정 필요.

#### 4. IK 함수 추가 - Jacobian / Mass Matrix (#1539)
**커밋**: `0fb91a4`

**새 Public API** (`newton.__init__.py`에 export):
- `newton.eval_jacobian()` - Articulation Jacobian 행렬 계산
- `newton.eval_mass_matrix()` - Articulation Mass 행렬 계산

```python
import newton

jacobian = newton.eval_jacobian(model, state, ...)
mass_matrix = newton.eval_mass_matrix(model, state, ...)
```

- `newton/_src/sim/articulation.py`에 대규모 구현 추가 (+466줄)
- `newton/_src/utils/selection.py` 기능 확장
- 새 테스트: `test_jacobian_mass_matrix.py` (+590줄)

**영향**: 새 기능 추가. 기존 코드에 영향 없음.

#### 5. Mimic Constraints 지원 준비 (#1523)
**커밋**: `d236161`

**새 API**: `ModelBuilder.add_constraint_mimic()` 메서드

URDF mimic 시맨틱스: `joint0 = coef0 + coef1 * joint1`

**새 Model 속성들**:
- `Model.constraint_mimic_joint0` / `joint1` - follower/leader 조인트 인덱스
- `Model.constraint_mimic_coef0` / `coef1` - offset/scale 계수
- `Model.constraint_mimic_enabled` / `_key` / `_world` / `_count`

**영향**: 새 기능 추가. 기존 코드에 영향 없음.

#### 6. Broad Phase 필터링 개선 (#1554)
**커밋**: `4484d7f`

- NxN 및 SAP broad phase가 `shape_collision_filter_pairs`를 올바르게 적용
- `CollisionPipeline.__init__()`에 `shape_pairs_excluded` 파라미터 추가
- `is_pair_excluded()` warp 함수로 바이너리 서치 기반 필터링

**영향**: Broad phase collision detection의 정확성 향상. 기존 코드에 영향 없음.

#### 7. Linesearch 기본값 변경: parallel → iterative (#1573)
**커밋**: `888069b`

- MuJoCo solver의 `ls_parallel` 기본값이 `True` → `False`로 변경
- 시뮬레이션 결정성(determinism) 향상
- 명시적으로 `ls_parallel=True` 설정은 여전히 가능

**영향**: MuJoCo solver 사용 시 기본 동작이 더 결정적으로 변경됨.

#### 8. Non-articulated Joints 임포트 지원 (#1535)
**커밋**: `f84449d`

- USD 임포트 시 orphan joints (articulation에 속하지 않는 조인트) 처리 지원
- Orphan joint 감지 시 경고 출력
- `finalize(skip_validation_joints=True)`로 검증 스킵 가능
- Maximal-coordinate solver에서만 지원

**영향**: 이전에 지원되지 않던 USD 에셋 임포트 가능.

#### 9. `mesh_maxhullvert` 파라미터 기본값 지연 해석 (#1587)
**커밋**: `b5ef434`

- `parse_urdf()`, `parse_mjcf()`, `parse_usd()`, `Mesh.__init__()` 등에서 `mesh_maxhullvert` 기본값이 `64` → `None`으로 변경
- 런타임에 `Mesh.MAX_HULL_VERTICES`로 해석됨
- 기능적으로 동일하나 시그니처 변경

#### 10. 버그 수정

- **Control.clear() 수정** (#1602, `019526b`): 잘못된 import로 인한 런타임 에러 수정
- **SolverMuJoCo OOB 읽기 수정** (#1607, `3af9c06`): 이종 월드에서 `geom_margin` 배열 범위 초과 수정
- **ViewerRerun hidden 파라미터 수정** (#1555, `b54e33b`): `log_mesh()`/`log_instances()`에서 `hidden` 파라미터가 무시되던 문제 수정
- **Child shape 필터링 수정** (#1559, `2c3df15`): 새 shape 추가 시 자식 body의 충돌 필터 누락 수정
- **SDF geometry finalize TypeError 수정** (#1586, `5f84fe1`): SDF geometry에 `device` kwarg 전달 시 TypeError 수정
- **example_robot_anymal_c_walk 수정** (#1574, `b64ff5a`): 충돌 파이프라인 호환성 문제 임시 해결
- **충돌 파이프라인 비결정성 수정** (#1588, `9e13713`): anymal_c_walking 예제의 비결정적 동작 수정

#### 11. 기타 개선

- **`--quiet` 플래그 추가** (#1585, `435e291`): 예제 실행 시 Warp 메시지 억제
- **USD schema resolver 개선** (#1463, `7d99b80`): MuJoCo solver 속성 스키마 리졸버 통합
- **CI 개선** (#1570, `6e8ee2f`): API docs 변경 감지 CI 체크 추가
- **Pillow 업데이트** (#1612, `5e6bf94`): 12.0.0 → 12.1.1
- **Joint custom attributes 중복 제거** (#1584, `e267b05`): 내부 리팩토링

---

## 2026-02-09 업데이트

### 커밋 범위
`8c7f116..2fcc770` (main branch merge)

### 주요 변경 사항

#### 1. Default shape_ke 변경 (#1491)
**커밋**: `0e5438f`

**중요**: `ModelBuilder.ShapeConfig.ke` 기본값이 변경됨.

| 항목 | Old | New |
|------|-----|-----|
| `ke` (contact elastic stiffness) | `1.0e3` | `2.5e3` |

**MJCF 파싱 개선**:
- `geom solref` 속성에서 contact stiffness/damping 파싱 지원

**영향**: 기존 시뮬레이션의 접촉 동작이 달라질 수 있음. 명시적으로 `ke` 값을 설정하지 않은 경우 확인 필요.

#### 2. Cable Junctions 지원 (#1519)
**커밋**: `b31d6d2`

**새 기능**:
- Cable Y-junction 지원
- `add_rod()` quaternions 파라미터가 optional로 변경 (자동 계산)
- 새 유틸리티: `newton/_src/utils/cable.py`

**새 예제**:
- `newton/examples/cable/example_cable_y_junction.py`

**API 변경**:
```python
# Old: quaternions 필수
builder.add_rod(positions, quaternions, ...)

# New: quaternions 자동 계산 가능
builder.add_rod(positions, radius=0.1, ...)  # quaternions=None
```

#### 3. VBD Solver 대규모 업데이트 (#1479)
**커밋**: `8e38779`

**새 기능**:
- Particle VBD kernels 대폭 개선
- Graph coloring 알고리즘 리팩토링
- Cloth, softbody 시뮬레이션 성능 향상

**새 예제들**:
- `example_rolling_cloth.py` - 천 롤링
- `example_falling_gift.py` - 선물 낙하 (multiphysics)
- `example_poker_cards_stacking.py` - 카드 쌓기
- `example_softbody_dropping_to_cloth.py` - 소프트바디-천 상호작용
- `example_softbody_hanging.py` - 매달린 소프트바디

#### 4. Viewer log_shapes 수정 (#1550)
**커밋**: `ccb0a89`

- Length-1 warp array broadcasting 버그 수정
- `viewer.log_shapes()` 사용 시 단일 shape 렌더링 문제 해결

#### 5. SolverMuJoCo 수정 (#1546)
**커밋**: `c996047`

- `update_solver_options_kernel`의 tolerance clamping 버그 수정
- Solver options 동적 업데이트 안정성 개선

#### 6. Warp Raytrace 추가 수정 (#1545)
**커밋**: `2dca881`

- 누락된 함수 호출에 device parameter 추가

#### 7. 문서화 개선 (#1560, #1566)
- Versioned documentation deployment to GitHub Pages
- 문서 링크 수정

---

## 2026-02-06 업데이트

### 커밋 범위
`08ea9bb..8c7f116` (main branch merge)

### 주요 변경 사항

#### 1. SensorTiledCamera API 변경 (#1516)
**커밋**: `d435c41`

Constructor에서 `width`, `height`, `num_cameras` 파라미터가 제거되고, 각 output 생성 메서드로 이동됨.

| 메서드 | Old API | New API |
|--------|---------|---------|
| `__init__` | `(model, num_cameras, width, height, options)` | `(model, options)` |
| `compute_pinhole_camera_rays` | `(fov)` | `(width, height, fov)` |
| `create_depth_image_output` | `()` | `(width, height, num_cameras)` |
| `create_color_image_output` | `()` | `(width, height, num_cameras)` |

**Output Shape 변경**:
- Old: `(total_pixels,)` 또는 `(num_worlds * num_cameras * height * width)`
- New: `(num_worlds, num_cameras, height, width)` - 4D array

**영향받는 파일**:
- `playground/experiments/franka_allegro_grasp/env.py`
- `playground/experiments/franka_allegro_grasp/view_depth.py`
- `playground/experiments/franka_allegro_grasp/visualize.py`

#### 2. API Refactor (#1481)
**커밋**: `958e363`

**제거된 함수들** (더 이상 `newton` 모듈에서 export되지 않음):
- `newton.count_rigid_contact_points`
- `newton.get_joint_constraint_count`
- `newton.get_joint_dof_count`
- `newton.infer_actuator_mode`

**추가된 모듈**:
- `newton.math` - 수학 유틸리티 모듈

#### 3. ModelBuilder 확장 (#1438, #1458)
- Per-world entity start-index arrays 추가
- Custom attributes에 `str` dtype 지원
- SolverMuJoCo custom attributes for solver options

#### 4. Hydroelastic Contacts Refactor (#1513)
- `contact_reduction_hydroelastic.py` 추가
- Contact reduction 로직 리팩토링

#### 5. MJCF Import 개선 (#1504)
- Actuator `*limited` flags 자동 활성화 (when `*range` specified)

#### 6. Texture/Material 지원 (#1393)
- Visual meshes에 texture, material loading 지원

#### 7. Warp Raytrace 개선 (#1542, #1544)
- Device parameter 추가
- Minor refactoring

---

## 업데이트 확인 방법

```bash
# 최신 커밋 확인
git log --oneline -20

# 특정 기간 변경사항 확인
git diff HEAD~N --stat

# 특정 파일 변경사항 확인
git diff HEAD~N..HEAD -- path/to/file
```

---

## Playground 호환성 체크리스트

업데이트 시 확인할 항목:

- [ ] `CollisionPipelineUnified` → `CollisionPipeline` 마이그레이션 (제거됨)
- [ ] `I_m` → `inertia` 파라미터 이름 변경 확인
- [ ] `MESH_MAXHULLVERT` → `Mesh.MAX_HULL_VERTICES` 변경 확인
- [ ] `SensorTiledCamera` API 호환성
- [ ] `SensorContact` API 호환성
- [ ] `ModelBuilder` API 호환성
- [ ] `ModelBuilder.ShapeConfig.ke` 기본값 변경 확인 (1.0e3 → 2.5e3)
- [ ] `add_rod()` API 변경 확인 (quaternions optional)
- [ ] `newton.*` 함수 export 확인 (`eval_jacobian`, `eval_mass_matrix` 추가)
- [ ] Solver API 변경 확인 (`ls_parallel` 기본값 변경)
- [ ] VBD solver 사용 시 그래프 컬러링 변경 확인
- [ ] 예제 실행 테스트
