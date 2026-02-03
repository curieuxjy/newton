# Newton 시작하기

## 날짜:

## 오늘 배운 것

### 기본 구조

Newton 시뮬레이션의 기본 흐름:

1. `ModelBuilder`로 모델 구성
2. 링크(link)와 셰이프(shape) 추가
3. 조인트(joint)로 연결
4. `Articulation`으로 묶기
5. `finalize()`로 모델 확정
6. 솔버 생성 및 시뮬레이션 루프

### 주요 함수들

```python
# 모델 빌더
builder = newton.ModelBuilder()

# 링크 (강체) 추가
link = builder.add_link()

# 셰이프 추가 (박스, 구, 캡슐 등)
builder.add_shape_box(link, hx=1.0, hy=0.1, hz=0.1)
builder.add_shape_sphere(link, radius=0.5)
builder.add_shape_capsule(link, radius=0.1, half_height=0.5)

# 조인트 추가
builder.add_joint_revolute(...)  # 회전 조인트
builder.add_joint_prismatic(...)  # 슬라이딩 조인트
builder.add_joint_fixed(...)  # 고정 조인트

# 바닥 추가
builder.add_ground_plane()

# 모델 확정
model = builder.finalize()
```

### 솔버 종류

- `SolverXPBD`: Extended Position-Based Dynamics
- (더 있을 수 있음 - 확인 필요)

## 질문/궁금한 점

-

## 다음에 해볼 것

- [ ] 다른 예제 실행해보기
- [ ] 로봇 예제 살펴보기
- [ ] 천(cloth) 시뮬레이션 이해하기
