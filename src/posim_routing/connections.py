"""Compile shorter port connectors against OSM land polygons, never a depth claim.

The global itinerary is a guide. Resident numeric terrain still determines every
native command. Existing passages, arrival centres and preserved prefixes remain
fixed; an unavailable coast connection leaves the existing course unchanged.
"""
import argparse
import hashlib
import heapq
import json
import math
import time
import tempfile
import shutil
import zipfile
from pathlib import Path

import numpy as np
import pyogrio
import shapely
from shapely.ops import transform
from pyproj import CRS, Geod, Transformer

GEO = Geod(ellps='WGS84')
POLICY = 'osm-port-connectors-v1'
SOURCE = 'https://osmdata.openstreetmap.de/download/land-polygons.html'


def length(points):
    return sum(GEO.inv(*a, *b)[2] for a, b in zip(points, points[1:]))


class CoastConnector:
    def __init__(self, archive, step=250., margin=80., timeout=10.):
        self.archive, self.step, self.margin, self.timeout = archive, step, margin, timeout
        self._cache = tempfile.TemporaryDirectory(prefix='www-posim-coast-')
        with zipfile.ZipFile(archive) as source:
            self.member = next(name for name in source.namelist() if name.endswith('.shp'))
            stem = self.member[:-4]
            # Extract once: repeated random reads inside this large compressed
            # shapefile otherwise inflate the archive for every connection.
            for suffix in ('.shp', '.shx', '.dbf', '.prj', '.cpg', '.qix'):
                member = stem+suffix
                if member not in source.namelist():continue
                destination = Path(self._cache.name)/Path(member).name
                with source.open(member) as incoming, destination.open('wb') as outgoing:
                    shutil.copyfileobj(incoming, outgoing)
        self.dataset = Path(self._cache.name)/Path(self.member).name

    def route(self, start, end):
        if abs(start[0]-end[0]) > 180:
            raise RuntimeError('date_line_connection_requires_review')
        bounds = (min(start[0], end[0])-.2, min(start[1], end[1])-.2,
                  max(start[0], end[0])+.2, max(start[1], end[1])+.2)
        metadata, _, geometries, _ = pyogrio.raw.read(
            str(self.dataset), bbox=bounds, columns=[])
        if metadata['crs'] != 'EPSG:4326':
            raise RuntimeError('unexpected_coast_projection')
        land = shapely.union_all(shapely.intersection(shapely.from_wkb(geometries), shapely.box(*bounds)))
        local = CRS.from_proj4(f'+proj=aeqd +lat_0={end[1]} +lon_0={end[0]} +datum=WGS84 +units=m')
        forward = Transformer.from_crs(4326, local, always_xy=True)
        inverse = Transformer.from_crs(local, 4326, always_xy=True)
        points = self.plan(transform(forward.transform, land), forward.transform(*start), forward.transform(*end),
                           transform(forward.transform, shapely.box(*bounds)))
        return [list(start), *[list(inverse.transform(*p)) for p in points[1:-1]], list(end)]

    def plan(self, land, start, end, coverage=None):
        barrier = land.buffer(self.margin)
        shapely.prepare(barrier)
        def visible(a, b):
            line = shapely.LineString([a, b])
            return not barrier.intersects(line) and (coverage is None or coverage.covers(line))
        if barrier.intersects(shapely.MultiPoint([start, end])):
            raise RuntimeError('endpoint_inside_coast_margin')
        if visible(start, end):
            return [start, end]
        step = self.step
        low = np.minimum(start, end)-15000.
        high = np.maximum(start, end)+15000.
        nx, ny = np.ceil((high-low)/step).astype(int)+1
        if nx*ny > 500000:
            raise RuntimeError('connection_grid_limit')
        xx, yy = np.meshgrid(low[0]+np.arange(nx)*step, low[1]+np.arange(ny)*step)
        # Enlarging blocked nodes encloses every connecting grid edge.
        blocked = shapely.intersects_xy(barrier.buffer(step/math.sqrt(2)), xx, yy)
        if coverage is not None:
            blocked |= ~shapely.contains_xy(coverage.buffer(-step/math.sqrt(2)), xx, yy)
        def xy(node):
            return low+np.asarray(node)*step
        def near(point):
            cell = np.rint((point-low)/step).astype(int)
            return [tuple(cell+(i, j)) for i in range(-2, 3) for j in range(-2, 3)
                    if 0 <= cell[0]+i < nx and 0 <= cell[1]+j < ny
                    and not blocked[cell[1]+j, cell[0]+i] and visible(point, xy(cell+(i, j)))]
        goals = set(near(end))
        heap, costs, parents, closed = [], {}, {}, set()
        for node in near(start):
            costs[node] = math.dist(start, xy(node))
            heapq.heappush(heap, (costs[node]+math.dist(xy(node), end), costs[node], node))
        deadline = time.monotonic()+self.timeout
        best, finish = math.inf, None
        while heap:
            estimate, cost, node = heapq.heappop(heap)
            if estimate >= best:
                break
            if node in closed:
                continue
            if time.monotonic() > deadline:
                raise RuntimeError('connection_search_budget')
            closed.add(node)
            if node in goals:
                candidate = cost+math.dist(xy(node), end)
                if candidate < best:
                    best, finish = candidate, node
            for i, j in ((-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (1, -1), (-1, 1), (1, 1)):
                nxt = (node[0]+i, node[1]+j)
                if not (0 <= nxt[0] < nx and 0 <= nxt[1] < ny) or blocked[nxt[1], nxt[0]]:
                    continue
                value = cost+step*math.hypot(i, j)
                if value >= costs.get(nxt, math.inf):
                    continue
                costs[nxt], parents[nxt] = value, node
                heapq.heappush(heap, (value+math.dist(xy(nxt), end), value, nxt))
        if finish is None:
            raise RuntimeError('no_coast_connection')
        raw = [end, xy(finish)]
        while finish in parents:
            finish = parents[finish]
            raw.append(xy(finish))
        raw.append(start)
        raw.reverse()
        result, index = [raw[0]], 0
        while index < len(raw)-1:
            last = len(raw)-1
            while not visible(raw[index], raw[last]):
                last -= 1
            result.append(raw[last])
            index = last
        if not all(visible(a, b) for a, b in zip(result, result[1:])):
            raise RuntimeError('connector_crosses_land')
        return result


def shorten(points, connector, protected=(), minimum_index=0):
    """Replace only an endpoint tail, and only with a shorter coast-checked one."""
    best, notes = points, []
    for index in range(max(1, len(points)-6, minimum_index), len(points)-1):
        if any(point in protected for point in points[index+1:]):
            continue
        distance = GEO.inv(*points[index], *points[-1])[2]
        if not 1000. < distance < 250000.:
            continue
        # The direct distance bounds the best possible connector.
        if length(points[:index+1])+distance >= length(best)-1000.:
            continue
        try:
            tail = connector.route(points[index], points[-1])
        except RuntimeError as error:
            notes.append(dict(index=index, reason=str(error)))
            continue
        candidate = points[:index]+tail
        if length(candidate) < length(best)-1000.:
            best = candidate
    return best, notes
