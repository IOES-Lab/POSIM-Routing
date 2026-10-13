"""Checked outward migration when a hull is already inside a shallow buffer."""
import math
import numpy as np
from scipy.ndimage import distance_transform_edt
from scipy.interpolate import RegularGridInterpolator

MINIMUM_DEPTH_M=2.
SHALLOW_BUFFER_M=200.


def clearance_field(field):
    dx,dy=float(np.max(np.diff(field.xs))),float(np.max(np.diff(field.ys)))
    shallow=~np.isfinite(field.heights) | (field.heights > -MINIMUM_DEPTH_M)
    distances=distance_transform_edt(~shallow,sampling=(dy,dx))
    return RegularGridInterpolator((field.ys,field.xs),distances,
                                  bounds_error=False,fill_value=0.)


def increasing_clearance(clearance,points):
    """An outward exception can only leave the existing shallow exclusion."""
    previous=None
    for a,b in zip(points,points[1:]):
        count=max(2,math.ceil(math.dist(a[:2],b[:2])/10.)+1)
        samples=np.linspace(a[:2],b[:2],count)
        values=clearance(samples[:,::-1])
        for value in values:
            if previous is not None and value < previous-1.:return False
            previous=float(value)
    return True


def exit_corridor(terrain,spec,start,target,guard=None):
    from posim_terrain.terrain import projection
    from .navigation import TerrainField,DetourPlanner
    from .planning import CachedTerrain
    from .land import LandExcludedField
    _,forward,inverse=projection(spec)
    raw=TerrainField(terrain['xs'],terrain['ys'],terrain['heights'])
    field=CachedTerrain(raw)
    if guard is not None:field=LandExcludedField(field,guard)
    origin=[*forward.transform(*start),0.];goal=np.array(forward.transform(*target))
    planner=DetourPlanner(field,origin,True,maximum_radius=2000,maximum_length=3000)
    planner.radius=SHALLOW_BUFFER_M+2*max(float(np.max(np.diff(raw.xs))),float(np.max(np.diff(raw.ys))))
    planner.clearance=MINIMUM_DEPTH_M
    if planner.safe(origin[:2],0.):return None
    # Migration cannot teleport a vessel that was already within 200 m.
    # Its bounded outward segment must remain hull-safe and increase separation.
    planner.radius=40.
    if not planner.safe(origin[:2],0.):raise RuntimeError('shallow_exit_hull_unsafe')
    clearance=clearance_field(raw);bearing=math.atan2(goal[1]-origin[1],goal[0]-origin[0])
    for travel in (100.,150.,200.,250.,350.,500.,750.,1000.):
        candidates=[]
        for angle in range(0,360,15):
            theta=bearing+math.radians(angle)
            end=[origin[0]+travel*math.cos(theta),origin[1]+travel*math.sin(theta),0.]
            planner.radius=SHALLOW_BUFFER_M+2*max(float(np.max(np.diff(raw.xs))),float(np.max(np.diff(raw.ys))))
            if not planner.safe(end[:2],0.):continue
            planner.radius=40.
            if not increasing_clearance(clearance,[origin,end]) or not planner.segment(origin,end):continue
            progress=float(np.linalg.norm(goal-origin[:2])-np.linalg.norm(goal-end[:2]))
            candidates.append((progress,end))
        if candidates:
            _,end=max(candidates,key=lambda value:value[0])
            return dict(points=[list(inverse.transform(*end[:2]))],index=0,mode='shallow_exit',
                shallow_water_buffer_m=SHALLOW_BUFFER_M,required_depth_m=MINIMUM_DEPTH_M,
                escaping_shallow_buffer=True,maximum_exit_length_m=travel,
                exit_start_clearance_m=float(clearance([[origin[1],origin[0]]])[0]),
                scope='hull-checked propulsion exit with increasing shallow-water separation')
    raise RuntimeError('shallow_exit_unavailable')
