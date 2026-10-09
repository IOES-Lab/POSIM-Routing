"""Mapped land is an exclusion, never evidence of water depth."""
import json
import math
import os
import zipfile
from functools import lru_cache
from pathlib import Path

import numpy as np
import shapely
from shapely.ops import transform


class LandGuard:
    def __init__(self, land, coverage):
        if not land.is_valid or not coverage.is_valid:
            raise RuntimeError('invalid_land_exclusion')
        self.land, self.coverage = land, coverage
        shapely.prepare(self.land)
        shapely.prepare(self.coverage)

    def free(self, x, y, radius):
        footprint = shapely.box(x-radius, y-radius, x+radius, y+radius)
        return self.coverage.covers(footprint) and not self.land.intersects(footprint)

    def segment(self, a, b, radius):
        # Enclose the swept square used by numeric triangle clearance.
        footprint = shapely.MultiPoint([(point[0]+dx, point[1]+dy)
                    for point in (a,b) for dx,dy in
                    ((-radius,-radius),(-radius,radius),(radius,radius),(radius,-radius))]).convex_hull
        return self.coverage.covers(footprint) and not self.land.intersects(footprint)

    def grid(self, xs, ys, radius=0.):
        xx, yy = np.meshgrid(xs, ys)
        margin = radius*math.sqrt(2)
        return (shapely.contains_xy(self.coverage.buffer(-margin), xx, yy)
                & ~shapely.intersects_xy(self.land.buffer(margin), xx, yy))


class LandExcludedField:
    def __init__(self, field, guard):
        self.field, self.guard = field, guard
        base = getattr(field, 'field', field)
        self.xs, self.ys, self.heights = base.xs, base.ys, base.heights

    def maximum(self, x, y, radius):
        if not self.guard.free(x, y, radius):
            return None
        return self.field.maximum(x, y, radius)


def guard_from_spec(spec, extent=None):
    from posim_terrain.terrain_sources import cache_root
    archive = Path(os.getenv('WWOS_COAST_OSM_ARCHIVE') or cache_root()/'coastal-land-polygons.zip')
    if not archive.is_file():
        raise RuntimeError('land_exclusion_data_unavailable')
    signature = (archive.stat().st_size, archive.stat().st_mtime_ns)
    return _load(json.dumps(spec, sort_keys=True), str(archive), signature, extent)


def geographic_windows(inverse,half):
    # Sample the perimeter in the actual floating frame. Unwrap around its
    # centre before splitting queries at the international date line.
    edges=np.linspace(-half,half,33)
    x=np.concatenate((edges,edges,np.full(33,-half),np.full(33,half)))
    y=np.concatenate((np.full(33,-half),np.full(33,half),edges,edges))
    lons,lats=inverse.transform(x,y);centre=inverse.transform(0.,0.)[0]
    lons=(np.asarray(lons)-centre+180)%360-180+centre
    lo,hi=float(lons.min())-.01,float(lons.max())+.01
    south,north=float(np.min(lats))-.01,float(np.max(lats))+.01
    if hi-lo>=180 or south < -85 or north>85:
        raise RuntimeError('land_exclusion_horizon_unavailable')
    return [(max(-180.,lo+shift),south,min(180.,hi+shift),north)
            for shift in (-360.,0.,360.) if lo+shift<180 and hi+shift>-180]


@lru_cache(maxsize=12)
def _load(encoded, archive, signature, extent):
    import pyogrio.raw
    from posim_terrain.terrain import projection
    spec = json.loads(encoded)
    _, forward, inverse = projection(spec)
    half = max(spec['width_m'], spec['height_m'])/2 + 500 if extent is None else extent
    windows=geographic_windows(inverse,half)
    with zipfile.ZipFile(archive) as source:
        member = next(name for name in source.namelist() if name.endswith('.shp'))
    parts=[]
    for bounds in windows:
        metadata,_,geometries,_=pyogrio.raw.read('/vsizip/'+archive+'/'+member,
                                               bbox=bounds,columns=[])
        if metadata['crs']!='EPSG:4326':
            raise RuntimeError('land_exclusion_projection_mismatch')
        try:
            land=shapely.union_all(shapely.intersection(shapely.from_wkb(geometries),shapely.box(*bounds)))
            parts.append(transform(forward.transform,land))
        except shapely.GEOSException as error:
            raise RuntimeError('invalid_land_exclusion_source') from error
    # Both geographic windows cover this local square; unioning their
    # projected geographic boundaries can leave a sub-nanometre wrap seam.
    return LandGuard(shapely.union_all(parts),shapely.box(-half,-half,half,half))
