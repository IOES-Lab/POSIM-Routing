# POSIM-Routing

[English](README.md)

세계 항로·수치 지형을 검사한 해안 경로·항구 접근 경로를 생성합니다. 전역 그래프로 방문 순서를 정하고, 수치 지형으로 지역·로컬 경로를 확인합니다. WWW-POSIM은 이 계획기를 지형 작업자와 항해 관리기에 연결합니다.

## 설치

Python 3.10 이상이 필요합니다. 지형 라이브러리와 함께 설치합니다.

```sh
git clone https://github.com/IOES-Lab/POSIM-Routing.git
cd POSIM-Routing
python -m venv .venv
. .venv/bin/activate
python -m pip install git+https://github.com/IOES-Lab/POSIM-Terrain.git@v0.1.0 -e .
```

Windows에서는 `.venv\Scripts\activate`로 가상환경을 활성화합니다.

## 세계 항로

좌표는 WGS84의 `[경도, 위도]`입니다.

```sh
python -m posim_routing --start 129.08468 35.07446 --end 139.83 35.60 --output out/busan-tokyo.geojson
```

```python
from posim_routing.global_route import route

course = route([129.08468, 35.07446], [139.83, 35.60])
```

전역 항로는 searoute 1.4.3을 사용하며 기본값으로 파나마와 북서항로를 제외합니다. 다른 방문 경로에는 `restrictions`와 `passage_gates`를 지정합니다. 항구 목록 좌표와 수역의 도착 목표는 별도 입력입니다. 전역 항로만으로 수심을 확인할 수는 없습니다.

## 생성 지형으로 로컬 경로 계획

```python
import numpy as np
from posim_routing.navigation import TerrainField, DetourPlanner

with np.load("out/busan/projected.npz") as grid:
    field = TerrainField(grid["xs"], grid["ys"], grid["elevation"])
start = [0., 0., 0.]
goal = [100., 0., 0.]
planner = DetourPlanner(field, start, boat=True)
path = planner.route(start, goal)
```

확인된 수역 안에서 출발점과 목표를 선택합니다. 자료가 없는 범위는 통과할 수 없습니다. 차량이 지나가는 전체 폭을 삼각형 충돌 지형과 비교합니다.

| 기준 | 기본값 |
|---|---|
| 수상 차량 로컬 수심 여유 | 1.5 m |
| 수상 차량 반경 + 추적 여유 | 3.5 m + 4 m |
| 해안 선호 거리 | 250 m |
| 해안 지역 여유 폭 / 수심 | 40 m / 2 m |
| 도착 반경 | 300 m |
| 도착 목표 검사 | 반경 350 m / 수심 3 m |

`arrival_targets.candidate(folder, port, radius=350., coast=None)`는 생성한 지형에서 수상 도착 목표를 고릅니다. 선택적으로 `connections.CoastConnector`를 전달하면 OSM 육지 형상으로 검사 범위 전체를 확인합니다. 기본 검사 범위는 반경 300 m 도착 원과 바깥 여유 50 m를 포함합니다.

`port_approach.corridor(tiles, spec, start, target)`는 도착 영역으로 들어가는 가능한 경로를 비교합니다. `connections.CoastConnector`는 OSM 육지 형상으로 전역 항로와 항구 사이의 연결을 단축합니다. 이 검사는 해안선 여유만 확인하며 수심은 확인하지 않습니다.

항구를 지나쳐 돌아오는 경로를 줄이기 위해 앞쪽 외해 경유점도 연결 후보로 검사합니다.
필수 통과점과 보존하도록 지정한 경로 구간은 유지합니다. 탐색에 실패하면 입력 경로를
유지합니다. 실제 주행에는 수치 지형 검사가 필요합니다.

- 후보 경유점에서 항구까지의 거리: 1–500 km
- 해안 탐색 격자 한도: 1,000,000개
- 기본 격자 간격: 250 m
- 기본 육지 여유 거리: 80 m
- 기본 탐색 제한 시간: 10초

## 구조

| 경로 | 내용 |
|---|---|
| `src/posim_routing/global_route.py` | 해양 그래프와 필수 통과점 |
| `connections.py` | 해안을 고려한 항구 연결 |
| `navigation.py`, `planning.py` | 충돌 지형·차량 폭을 고려한 A*·지형 캐시 |
| `coastal.py` | 해안 탐색과 수치 지형 경로 |
| `regional.py` | 비동기 지역 경로 준비 |
| `arrival_targets.py` | 수치 지형과 선택적인 OSM 육지 검사로 도착 수역 선택 |
| `port_approach.py` | 도착 영역 접근과 곡선 감속 |
| `course.py`, `navigation_frame.py` | 항로 진행과 지리 좌표 전달 |
| `data/` | Natural Earth 해안선과 출처 |
| `tests/` | 해안·자료 범위·백그라운드 계획 검사 |

`RegionalNavigator`는 `/terrain/jobs`와 지형 작업 조회를 위한 비동기 `rpc(path, body=None, timeout=...)` 함수를 받습니다. 선택적인 `before_prepare` 함수로 호출 앱의 캐시를 관리합니다. Gazebo 실행·종료는 호출 앱이 담당합니다.

## 시험과 이용 조건

```sh
python -m unittest discover -s tests
```

코드: [Apache-2.0](LICENSE). [출처 고지](NOTICE.md)를 확인하세요. 경로는 시뮬레이션용이며 공인 항해 해도가 아닙니다.
