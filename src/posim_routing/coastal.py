"""Prefer visible coast without treating a cartographic shoreline as collision data.

Natural Earth only discovers regions worth querying. Numeric terrain determines
the shoreline, depth, and every regional corridor. The native Gazebo planner
independently approves each local corridor against its resident collision mesh.
"""
import gzip
import hashlib
import heapq
import json
import math
import time
from pathlib import Path

import numpy as np
from pyproj import Geod
from scipy.ndimage import distance_transform_edt, maximum_filter
from scipy.spatial import cKDTree

GEO = Geod(ellps="WGS84")
COAST_DISCOVERY_M = 80000.
STANDOFF_M = 250.
MAXIMUM_STANDOFF_M = 800.
MINIMUM_DEPTH_M = 2.
REGIONAL_BUFFER_M = 40.


def sphere(coordinates):
    radians = np.radians(coordinates)
    lon, lat = radians[..., 0], radians[..., 1]
    return np.stack((np.cos(lat)*np.cos(lon), np.cos(lat)*np.sin(lon), np.sin(lat)), axis=-1)


class CoastIndex:
    """Longitude-wrap-safe discovery index; never used to approve navigation."""
    def __init__(self, directory=None):
        directory = directory or Path(__file__).parent / "data"
        raw = gzip.decompress((directory / "coastline.json.gz").read_bytes())
        source = json.loads((directory / "coastline-source.json").read_text())
        if hashlib.sha256(raw).hexdigest() != source["sha256"]:
            raise RuntimeError("coastline_checksum_mismatch")
        lines = []
        for feature in json.loads(raw)["features"]:
            geometry = feature["geometry"]
            lines.extend([geometry["coordinates"]] if geometry["type"] == "LineString"
                         else geometry["coordinates"])
        points = []
        for line in lines:
            coordinates = np.asarray(line, dtype=float)[:, :2]
            points.append(coordinates)
            # A sparse cartographic segment must not hide a nearby coast.
            _, _, lengths = GEO.inv(coordinates[:-1, 0], coordinates[:-1, 1],
                                    coordinates[1:, 0], coordinates[1:, 1])
            for index in np.flatnonzero(lengths > 1000):
                points.append(np.asarray(GEO.npts(*coordinates[index], *coordinates[index+1],
                                                  math.ceil(lengths[index]/1000)-1)))
        self.points = np.concatenate(points)
        self.tree = cKDTree(sphere(self.points))

    def nearest(self, position):
        chord, index = self.tree.query(sphere(position))
        return float(2*6371008.8*math.asin(min(1., chord/2))), self.points[index].tolist()

    def progressive_shore(self, position, target):
        """Discover a nearby coast ahead, without approving any water depth.

        The nearest coastline can be behind a departing ship. Choose only
        coastline samples that advance towards the existing course waypoint;
        this never replaces port order or skips a course waypoint.
        """
        radius = 2*math.sin(COAST_DISCOVERY_M/(2*6371008.8))
        indices = self.tree.query_ball_point(sphere(position), radius)
        if not indices:
            return None
        points = self.points[indices]
        _, _, travel = GEO.inv(np.full(len(points), position[0]), np.full(len(points), position[1]),
                               points[:, 0], points[:, 1])
        _, _, remaining = GEO.inv(np.full(len(points), target[0]), np.full(len(points), target[1]),
                                  points[:, 0], points[:, 1])
        total = GEO.inv(*position, *target)[2]
        progress = total-remaining
        candidates = (travel < COAST_DISCOVERY_M) & (progress >= min(1000., total*.15))
        if not candidates.any():
            return None
        # Prefer nearby coast a little ahead rather than a distant headland.
        scores = np.where(candidates, travel + .2*np.abs(progress-7500.), math.inf)
        i = int(np.argmin(scores))
        return float(travel[i]), points[i].tolist()


def coastal_approach(start, target, shore):
    """One bounded step towards distant coast, still progressing on course.

    Numeric regional and native swept-hull checks must approve this candidate.
    It is a discovery target, never an assertion that the coast is navigable.
    """
    heading, _, remaining = GEO.inv(*start, *target)
    if remaining < 2000:
        return None
    bearing, _, separation = GEO.inv(*start, *shore)
    if separation <= 2500:
        return None
    # Add a forward component so sidewards approach does not undo progress.
    a, b = math.radians(heading), math.radians(bearing)
    east, north = .65*math.sin(b)+.35*math.sin(a), .65*math.cos(b)+.35*math.cos(a)
    angle = math.degrees(math.atan2(east, north))
    candidate = list(GEO.fwd(*start, angle, min(7500., remaining*.5))[:2])
    if remaining-GEO.inv(*candidate, *target)[2] < 500:
        return None
    return candidate


def coastal_corridor(terrain, spec, start, target, timeout=25.):
    """Return a progressive coastal path, or None to retain the maritime course.

    A bounded weighted A* prefers a 250 m stand-off. It cannot invent land or
    depth in missing cells, cross a headland, or pull an intermediate waypoint
    backwards. Close to a course waypoint, exact arrival takes precedence.
    """
    from posim_terrain.terrain import projection
    from .navigation import TerrainField, DetourPlanner
    from .planning import CachedTerrain

    started = time.monotonic()
    deadline = started + timeout
    _, forward, inverse = projection(spec)
    field = TerrainField(terrain["xs"], terrain["ys"], terrain["heights"])
    xs, ys, heights = field.xs, field.ys, field.heights
    finite = np.isfinite(heights)
    land = finite & (heights >= 0)
    if not land.any():
        return None
    origin = np.array(forward.transform(*start))
    destination = np.array(forward.transform(*target))
    remaining = float(np.linalg.norm(destination-origin))
    if remaining < 2000:
        return None
    dx, dy = float(np.max(np.diff(xs))), float(np.max(np.diff(ys)))
    if dx <= 0 or dy <= 0:
        raise RuntimeError("invalid_coastal_grid")
    # Cover the footprint, connecting triangles, and half a diagonal grid
    # edge. Unknown vertices remain impassable rather than becoming coastline.
    margin = REGIONAL_BUFFER_M + math.hypot(dx, dy)/2
    size = (2*math.ceil(margin/dy)+1, 2*math.ceil(margin/dx)+1)
    bounds = maximum_filter(np.where(finite, heights, math.inf), size=size,
                            mode="constant", cval=math.inf)
    safe = bounds <= -MINIMUM_DEPTH_M
    coast_distance = distance_transform_edt(~land, sampling=(dy, dx))
    xx, yy = np.meshgrid(xs, ys)
    travel = np.hypot(xx-origin[0], yy-origin[1])
    safe &= travel <= 18000
    to_target = np.hypot(xx-destination[0], yy-destination[1])
    progress = remaining-to_target
    coast = safe & (coast_distance >= 200) & (coast_distance <= MAXIMUM_STANDOFF_M)
    coast &= (travel >= 1000) & (travel <= 15000)
    if not (coast & (progress >= min(1000., remaining*.2))).any():
        return None
    # A near-coast endpoint can be the head of a closed bay. First establish
    # a through route to the onward target or a forward water exit, then take
    # its coastal prefix. Distance gain at an isolated endpoint is insufficient.
    goal_node = (int(np.argmin(abs(ys-destination[1]))), int(np.argmin(abs(xs-destination[0]))))
    target_inside = (xs[0]+margin < destination[0] < xs[-1]-margin
                     and ys[0]+margin < destination[1] < ys[-1]-margin
                     and safe[goal_node])
    edge = ((xx <= xs[0]+margin+2*dx) | (xx >= xs[-1]-margin-2*dx)
            | (yy <= ys[0]+margin+2*dy) | (yy >= ys[-1]-margin-2*dy))
    exits = safe & edge & (travel >= 1000) & (progress >= min(1000., remaining*.2))
    scores = np.where(exits, to_target, math.inf)
    goals = []
    if target_inside:
        goals.append(goal_node)
    for _ in range(0 if target_inside else 6):
        node = tuple(map(int, np.unravel_index(np.argmin(scores), scores.shape)))
        if not math.isfinite(float(scores[node])):
            break
        goals.append(node)
        scores[np.hypot(xx-xs[node[1]], yy-ys[node[0]]) < 600] = math.inf
    penalty = (1 + np.minimum(1,np.abs(coast_distance-STANDOFF_M)/STANDOFF_M)
               + 8*np.minimum(6,np.maximum(0,coast_distance-(STANDOFF_M+150))/500))
    validator = DetourPlanner(CachedTerrain(field), [*origin, 0.], True,
                             timeout=timeout, maximum_radius=18000, maximum_length=40000)
    validator.radius = REGIONAL_BUFFER_M
    validator.clearance = MINIMUM_DEPTH_M
    if not validator.safe(origin, 0.):
        return None

    def xy(node):
        return [float(xs[node[1]]), float(ys[node[0]]), 0.]

    start_node = (int(np.argmin(abs(ys-origin[1]))), int(np.argmin(abs(xs-origin[0]))))
    if not safe[start_node] or not validator.segment([*origin, 0.], xy(start_node)):
        return None
    expanded = 0
    offsets = ((0, 1), (0, -1), (1, 0), (-1, 0), (1, 1), (1, -1), (-1, 1), (-1, -1))
    for goal in goals:
        costs, parents = {start_node: 0.}, {}
        heap = [(math.dist(xy(start_node), xy(goal)), 0., start_node)]
        found = False
        while heap:
            if time.monotonic() > deadline:
                raise RuntimeError("coastal_planning_budget_exceeded")
            _, cost, node = heapq.heappop(heap)
            if cost != costs.get(node):
                continue
            expanded += 1
            if node == goal:
                found = True
                break
            for j, i in offsets:
                nxt = (node[0]+j, node[1]+i)
                if not (0 <= nxt[0] < len(ys) and 0 <= nxt[1] < len(xs)) or not safe[nxt]:
                    continue
                if i and j and (not safe[node[0]+j, node[1]] or not safe[node[0], node[1]+i]):
                    continue
                edge = math.hypot(i*dx, j*dy)*(penalty[node]+penalty[nxt])/2
                value = cost+edge
                if value >= costs.get(nxt, math.inf):
                    continue
                costs[nxt], parents[nxt] = value, node
                heapq.heappush(heap, (value+math.dist(xy(nxt), xy(goal)), value, nxt))
        if not found:
            continue
        nodes = [goal]
        while nodes[-1] != start_node:
            nodes.append(parents[nodes[-1]])
        nodes.reverse()
        coastal_nodes = [index for index,node in enumerate(nodes) if coast[node]]
        if not coastal_nodes:
            continue
        if not any(progress[nodes[index]] >= min(1000., remaining*.2) for index in coastal_nodes):
            continue
        through = [[*origin, 0.], *[xy(node) for node in nodes[1:]]]
        if sum(math.dist(a, b) for a, b in zip(through, through[1:])) > 40000:
            continue
        if not all(validator.segment(a,b,record=True) for a,b in zip(through,through[1:])):
            continue
        bands = {index: math.floor(abs(coast_distance[nodes[index]]-STANDOFF_M)/max(dx,dy))
                 for index in coastal_nodes}
        closest_band = min(bands.values())
        end = max(index for index in coastal_nodes if bands[index]==closest_band)
        onward_nodes = nodes[end+1:]
        endpoint = nodes[end]
        nodes = nodes[:end+1]
        # Shorten the grid path without erasing its coastal preference. Long
        # offshore shortcuts across bays have a higher coast-weighted cost.
        points, index = [[*origin, 0.]], 0
        while index < len(nodes)-1:
            chosen = index+1
            for end in range(min(len(nodes)-1, index+10), index, -1):
                a, b = xy(nodes[index]), xy(nodes[end])
                length = math.dist(a, b)
                count = max(2, math.ceil(length/min(dx, dy)*3))
                ix = np.rint(np.linspace(nodes[index][1], nodes[end][1], count)).astype(int)
                iy = np.rint(np.linspace(nodes[index][0], nodes[end][0], count)).astype(int)
                weighted = length*float(np.mean(penalty[iy, ix]))
                old = costs[nodes[end]]-costs[nodes[index]]
                if safe[iy, ix].all() and weighted <= old*1.08:
                    chosen = end
                    break
            points.append(xy(nodes[chosen]))
            index = chosen
        if sum(math.dist(a, b) for a, b in zip(points, points[1:])) > 40000:
            continue
        if not all(validator.segment(a, b, record=True) for a, b in zip(points, points[1:])):
            continue
        return dict(points=[list(inverse.transform(*point[:2])) for point in points[1:]],
                    index=0, mode="coastal", planning_seconds=time.monotonic()-started,
                    search_nodes=expanded, minimum_regional_clearance_m=validator.minimum,
                    preferred_standoff_m=STANDOFF_M,
                    endpoint_standoff_m=float(coast_distance[endpoint]),
                    maximum_preferred_standoff_m=MAXIMUM_STANDOFF_M,
                    progress_to_course_waypoint_m=float(progress[endpoint]),
                    onward_points=[list(inverse.transform(*xy(node)[:2])) for node in onward_nodes],
                    onward_exit=list(inverse.transform(*xy(goal)[:2])),
                    onward_exit_progress_m=float(progress[goal]),
                    onward_verified=True,
                    scope="numeric terrain coastal preference; native corridors checked separately")
    return None
