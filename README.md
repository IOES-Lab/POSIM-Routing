# POSIM-Routing

[한국어](README.ko.md)

Plan global maritime routes, coastal corridors and port approaches. A global graph supplies the itinerary. Numeric terrain checks depth, and mapped land excludes shore crossings. WWW-POSIM connects these planners to its terrain workers and voyage coordinator.

## Install

Python 3.10 or later is required. Install the terrain dependency and this library together.

```sh
git clone https://github.com/IOES-Lab/POSIM-Routing.git
cd POSIM-Routing
python -m venv .venv
. .venv/bin/activate
python -m pip install git+https://github.com/IOES-Lab/POSIM-Terrain.git@v0.1.0 -e .
```

On Windows, activate `.venv\Scripts\activate` instead.

## Global route

Coordinates are `[longitude, latitude]` in WGS84.

```sh
python -m posim_routing --start 129.08468 35.07446 --end 139.83 35.60 --output out/busan-tokyo.geojson
```

```python
from posim_routing.global_route import route

course = route([129.08468, 35.07446], [139.83, 35.60])
```

Global routes use searoute 1.4.3 and exclude Panama and the northwest passage by default. Set `restrictions` and `passage_gates` explicitly for another itinerary. Port catalogue coordinates and water arrival targets are separate inputs. A global route alone does not establish water depth.

## Local route from generated terrain

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

Choose start and goal inside confirmed water. The planner rejects unknown coverage and continuously checks the swept footprint against the piecewise planar collision field.

`land.LandExcludedField(field, guard)` adds mapped-land exclusion to a numeric
field. `land.guard_from_spec(spec)` reads the cached OSM land polygon archive.
Set `WWOS_COAST_OSM_ARCHIVE` to another local archive path. WWW-POSIM prepares
this dataset in its terrain provider cache. Missing data holds regional planning.
OSM land can reject a path; numeric terrain must still establish water depth.

Regional lookahead finds a complete mapped detour around a peninsula before
selecting a local depth-checked goal. The detour can initially increase distance
to the course target. At its checked endpoint, the caller waits for the next
checked segment. Longitude wrapping supports the international date line.

Coastal lookahead follows a checked water route to the onward target or a forward
exit from the regional grid. Its endpoint has an onward route around bays and
headlands. Ordinary coastal course vertices guide progress; ports and required
passage points retain their arrival checks.

Onward waypoints mark checked segment ends. Straight grid runs are shortened
when the complete swept corridor passes depth and mapped-land checks. The
shortened route preserves its exit and coastline preference.

| Policy | Default |
|---|---|
| Surface local depth clearance | 1.5 m |
| Surface hull radius + tracking margin | 3.5 m + 4 m |
| Coastal preferred distance | 250 m |
| Coastal regional buffer / depth | 40 m / 2 m |
| Arrival radius | 300 m |
| Arrival target screening | 350 m buffer / 3 m depth |

`arrival_targets.candidate(folder, port, radius=350., coast=None)` selects a water target from generated terrain. An optional `connections.CoastConnector` also checks the whole footprint against OSM land. The default footprint covers the 300 m arrival circle plus 50 m of margin.

`port_approach.corridor(tiles, spec, start, target)` compares feasible entries to the arrival region. `connections.CoastConnector` shortens graph-to-port connections using OSM land polygons; it checks coastline margins, not bathymetry.

The connector considers earlier offshore waypoints to avoid overshooting a port.
It retains required passage points and any preserved route prefix. A failed
search retains the supplied route. Numeric terrain checks remain necessary.

- Candidate distance to the port: 1–500 km
- Coast search grid limit: 1,000,000 nodes
- Default grid spacing: 250 m
- Default land margin: 80 m
- Default search timeout: 10 seconds

## Structure

| Path | Contents |
|---|---|
| `src/posim_routing/global_route.py` | Maritime graph and passage gates |
| `connections.py` | Coast-aware port connectors |
| `navigation.py`, `planning.py` | Collision field, swept-footprint A* and cached terrain |
| `coastal.py` | Coastal discovery and numeric corridors |
| `land.py`, `land_guidance.py` | Mapped-land exclusion and bounded around-land guidance |
| `regional.py` | Asynchronous regional lookahead |
| `arrival_targets.py` | Buffered numeric arrival water and optional OSM land checks |
| `port_approach.py` | Arrival-region corridors and bend braking |
| `course.py`, `navigation_frame.py` | Course progress and geographic handoff |
| `data/` | Natural Earth coastline and provenance |
| `tests/` | Coastal, coverage and background planning contracts |

`RegionalNavigator` accepts an async `rpc(path, body=None, timeout=...)` callback for `/terrain/jobs` and job reads. The optional `before_prepare` callback lets the caller manage its cache. It does not start or stop Gazebo.

Set `retain_coast=True` only for an ordinary shape advance within the same leg.
When `state['regional_waiting']` is present, the caller holds thrust until a
checked replacement is ready. Physics and sensor sampling continue during the
wait. Failed or empty optional renewal cannot replace a checked coastal route
with an offshore goal.

## Tests and terms

```sh
python -m unittest discover -s tests
```

Code: [Apache-2.0](LICENSE). See [source notices](NOTICE.md). Routes support simulation and are not certified navigation charts.
