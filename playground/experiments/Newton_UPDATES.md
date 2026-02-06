# Newton Simulator Updates Tracking

이 문서는 Newton 물리 시뮬레이터 메인 팀의 업데이트를 추적합니다.
**업데이트 요청 시에만 기록됩니다.**

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
- [ ] `newton.*` 함수 export 확인
- [ ] Solver API 변경 확인
- [ ] 예제 실행 테스트
