# Newton 학습 놀이터

원본 코드를 훼손하지 않고 Newton 시뮬레이터를 배우기 위한 개인 학습 공간입니다.

## 폴더 구조

```
playground/
├── README.md          # 이 파일
├── notes/             # 학습 노트
└── experiments/       # 실험용 예제 코드
```

## 예제 실행 방법

Newton 예제는 다음과 같이 실행합니다:

```bash
# 환경 설정 (처음 한 번만)
uv sync --extra examples

# 기본 예제 실행
uv run -m newton.examples basic_pendulum
uv run -m newton.examples basic_shapes
uv run -m newton.examples basic_joints

# 로봇 예제
uv run -m newton.examples robot_h1
uv run -m newton.examples robot_anymal_d

# 천(cloth) 시뮬레이션
uv run -m newton.examples cloth_hanging
```

## 나만의 실험 코드 실행

```bash
# playground 폴더의 스크립트 실행
uv run python playground/experiments/my_first_sim.py
```

## 학습 팁

1. **원본 예제 참고**: `newton/examples/` 폴더의 예제들을 참고하세요
2. **복사 후 수정**: 원본 예제를 `playground/experiments/`에 복사한 후 수정하세요
3. **노트 작성**: 배운 내용을 `playground/notes/`에 기록하세요

## 주요 개념

- `newton.ModelBuilder()`: 시뮬레이션 모델 구성
- `newton.solvers.SolverXPBD`: 물리 솔버
- `model.state()`: 시뮬레이션 상태 (위치, 속도 등)
- `model.collide()`: 충돌 감지

## 유용한 링크

- [Newton 문서](docs/)
- [예제 목록](../README.md)
