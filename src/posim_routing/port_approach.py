"""Final port approaches use resident collision terrain, not coastal scenery goals."""
import math
import json
import time
from functools import lru_cache

ARRIVAL_RADIUS_M = 300.
APPROACH_DISTANCE_M = 5000.


class ArrivalOutsideResidentTerrain(RuntimeError):
    """The vessel must advance before its arrival area is resident."""


@lru_cache(maxsize=8)
def forward_frame(spec):
    from posim_terrain.terrain import projection
    return projection(json.loads(spec))[1]


def bend_lookahead(native, regional, seed_spec):
    """Include the next regional bend even when the local corridor ends before it."""
    points=native.get('waypoints',[])
    if regional.get('mode')!='port_approach':return points
    spec=(native.get('origin') or {}).get('spec') or seed_spec
    forward=forward_frame(json.dumps(spec,sort_keys=True))
    upcoming=[[*forward.transform(*p),0.] for p in regional['points'][regional['index']:regional['index']+3]]
    if points and upcoming and math.dist(points[-1][:2],upcoming[0][:2])<2.:upcoming.pop(0)
    return [*points,*upcoming]


def shortest_arrival(planner, field, origin, candidates):
    """Compare feasible region entries within one shared planning budget."""
    best, best_floor, best_length = None, None, math.inf
    floors = [(goal, field.maximum(*goal[:2], 25.)) for goal in candidates]
    coverage = field
    while hasattr(coverage, 'field'):
        coverage = coverage.field
    if all(floor is None for _, floor in floors) and all(
            coverage.maximum(*goal[:2], 25.) is None for goal in candidates):
        raise ArrivalOutsideResidentTerrain('port_arrival_region_outside_resident_terrain')
    for goal, floor in sorted(floors, key=lambda item: math.dist(origin[:2], item[0][:2])):
        if math.dist(origin[:2], goal[:2]) >= best_length-.01:
            break
        if time.monotonic() > planner.deadline:
            break
        if floor is None or floor > -3.:
            continue
        try:
            path = planner.route(origin, goal)
        except RuntimeError:
            continue
        distance = sum(math.dist(a[:2], b[:2]) for a, b in zip([origin]+path[:-1], path))
        if distance < best_length:
            best, best_floor, best_length = path, floor, distance
    if best is None:
        raise RuntimeError('port_arrival_region_no_confirmed_corridor')
    return best, best_floor


def corridor(tiles, spec, start, target, guard=None):
    from posim_terrain.terrain import projection
    from .navigation import TerrainField, DetourPlanner
    from .planning import CachedTerrain
    _, forward, inverse = projection(spec)
    origin = [*forward.transform(*start), 0.]
    centre = [*forward.transform(*target), 0.]
    field = CachedTerrain(TerrainField.tiles(tiles))
    if guard is not None:
        from .land import LandExcludedField
        field = LandExcludedField(field, guard)
    planner = DetourPlanner(field, origin, True, step=8., timeout=20.,
                           maximum_radius=6000., maximum_length=16000.)
    # Preserve the catalog's 3 m / 25 m port endpoint screening. En route,
    # use exactly the live hull/tracking/depth rules, without inventing water.
    heading=math.atan2(origin[1]-centre[1],origin[0]-centre[0])
    candidates=[[*[centre[i]+250.*fn(heading+offset*math.pi/8) for i,fn in enumerate((math.cos,math.sin))],0.]
                for offset in (0,1,-1,2,-2,3,-3,4,-4,6,-6,8)] + [centre]
    path, floor = shortest_arrival(planner, field, origin, candidates)
    if not all(planner.segment(a, b, record=True)
               for a, b in zip([origin]+path[:-1], path)):
        raise RuntimeError('port_corridor_not_continuously_safe')
    return dict(points=[list(inverse.transform(*p[:2])) for p in path], index=0,
                mode='port_approach', minimum_regional_clearance_m=planner.minimum,
                required_depth_m=planner.clearance, regional_buffer_m=planner.radius,
                endpoint_buffer_m=25., endpoint_minimum_depth_m=-floor,
                arrival_radius_m=ARRIVAL_RADIUS_M,arrival_centre=list(target),
                scope='resident Gazebo collision field; every native command rechecked')


def manoeuvre_speed(cruise, position, points, index, terminal=False, remaining=math.inf):
    """Brake ahead of narrow bends, before the straight stopping guard fires."""
    speed = min(cruise, 4.) if terminal else cruise
    if terminal:
        speed = min(speed, math.sqrt(.8**2 + 2*.35*max(0., remaining-40.)))
    route = [position, *points[max(0, index):]]
    distance = 0.
    for a, b, c in zip(route, route[1:], route[2:]):
        incoming = [b[i]-a[i] for i in (0, 1)]
        outgoing = [c[i]-b[i] for i in (0, 1)]
        before, after = math.hypot(*incoming), math.hypot(*outgoing)
        distance += before
        if before < .01 or after < .01:
            continue
        bend = math.acos(max(-1., min(1., sum(x*y for x,y in zip(incoming,outgoing))/(before*after))))
        if bend > .35:
            turn_speed = .8 if bend > .6 else 2.
            speed = min(speed, math.sqrt(turn_speed**2 + 2*.35*max(0., distance-40.)))
    return speed
