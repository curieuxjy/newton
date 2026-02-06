@AGENTS.md

## Playground Experiments Guidelines

### Newton Updates Tracking

Newton 시뮬레이터 메인 팀의 업데이트를 추적하기 위해 `playground/experiments/Newton_UPDATES.md` 파일을 사용합니다.

**중요**: 이 파일은 사용자가 명시적으로 요청할 때만 업데이트합니다.

업데이트 추적 요청 시 수행할 작업:
1. `git log --oneline -20`으로 최신 커밋 확인
2. `git diff HEAD~N --stat`으로 변경된 파일 확인
3. 주요 API 변경사항 분석 (특히 sensors, sim, solvers 디렉토리)
4. `playground/experiments/Newton_UPDATES.md`에 변경사항 기록
5. 영향받는 playground 예제 파일 업데이트

주요 추적 대상:
- `newton/_src/sensors/` - SensorTiledCamera, SensorContact API
- `newton/_src/sim/` - ModelBuilder, Model, State API
- `newton/_src/solvers/` - Solver API 변경
- `newton/__init__.py` - Public API export 변경
