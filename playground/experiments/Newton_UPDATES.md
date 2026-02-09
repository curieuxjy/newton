# Newton Simulator Updates Tracking

이 문서는 Newton 물리 시뮬레이터 메인 팀의 업데이트를 추적합니다.
**업데이트 요청 시에만 기록됩니다.**

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

- [ ] `SensorTiledCamera` API 호환성
- [ ] `SensorContact` API 호환성
- [ ] `ModelBuilder` API 호환성
- [ ] `ModelBuilder.ShapeConfig.ke` 기본값 변경 확인 (1.0e3 → 2.5e3)
- [ ] `add_rod()` API 변경 확인 (quaternions optional)
- [ ] `newton.*` 함수 export 확인
- [ ] Solver API 변경 확인
- [ ] VBD solver 사용 시 그래프 컬러링 변경 확인
- [ ] 예제 실행 테스트
