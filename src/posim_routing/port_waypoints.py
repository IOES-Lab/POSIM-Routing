"""Mandatory, prechecked destination waypoints; native terrain still approves motion."""
import math,hashlib,json
from pyproj import Geod

GEO=Geod(ellps='WGS84')
POLICY='fixed-port-waypoints-v1'
WAYPOINT_RADIUS_M=5.
APPROACH_LENGTH_M=20000.
BUFFER_M=12.
DEPTH_M=1.5


def split_incoming(points, reach=APPROACH_LENGTH_M):
    """Split at an actual point on the incoming global polyline."""
    remaining=reach
    for index in range(len(points)-2,-1,-1):
        heading,_,length=GEO.inv(*points[index+1],*points[index])
        if length>=remaining:
            entry=list(GEO.fwd(*points[index+1],heading,remaining)[:2])
            return [*points[:index+1],entry],[entry,*points[index+1:]]
        remaining-=length
    return [list(points[0])],[list(p) for p in points]


def certificate(leg,index):
    departure=leg.get('fixed_departure') or {}
    if (departure.get('policy')==POLICY and departure.get('verified') is True
            and departure['first_waypoint']<=index<=departure['last_waypoint']):return departure
    approach=leg.get('fixed_approach') or {}
    if (approach.get('policy')==POLICY and approach.get('verified') is True
            and index>=approach['first_waypoint']):return approach
    return None


def mandatory(leg,index):return certificate(leg,index) is not None


def radius(leg,index):
    return WAYPOINT_RADIUS_M if mandatory(leg,index) else 50.


def subdivide(points, maximum=150.):
    """Retain all checked bends and add bounded native lookahead points."""
    result=[list(points[0])]
    for a,b in zip(points,points[1:]):
        heading,_,length=GEO.inv(*a,*b)
        count=max(1,math.ceil(length/maximum))
        result.extend([list(GEO.fwd(*a,heading,length*k/count)[:2]) for k in range(1,count)])
        result.append(list(b))
    return result


def remaining_points(leg,index):
    return leg['coordinates'][index:] if mandatory(leg,index) else None


def coordinates_hash(points):
    return hashlib.sha256(json.dumps(points,separators=(',',':')).encode()).hexdigest()


def attach_atlas(book,atlas):
    """Reapply checked approaches after rebuilding the same incoming course."""
    from .connections import length
    if atlas.get('policy')!=POLICY or set(atlas['ports'])!={p['id'] for p in book['ports']}:
        raise ValueError('fixed_port_atlas_itinerary_mismatch')
    verify_book(book)
    for leg in book['legs']:
        row=atlas['ports'][leg['destination']]
        if row['origin']!=leg['origin']:raise ValueError('fixed_port_atlas_origin_changed')
        if (coordinates_hash(leg['coordinates'])!=row['input_coordinates_sha256']
                and leg.get('global_course_coordinates_sha256')!=row['input_coordinates_sha256']
                and leg['coordinates']!=row['coordinates']):
            raise ValueError('fixed_port_global_course_changed:'+leg['destination'])
        leg['coordinates']=[list(p) for p in row['coordinates']]
        leg['fixed_approach']=dict(row['fixed_approach'])
        if row.get('fixed_departure'):leg['fixed_departure']=dict(row['fixed_departure'])
        leg['length_m']=length(leg['coordinates'])
    book['total_m']=sum(leg['length_m'] for leg in book['legs'])
    book['fixed_approaches']=dict(policy=POLICY,destinations=len(atlas['ports']),
        atlas_sha256=hashlib.sha256(json.dumps(atlas,sort_keys=True).encode()).hexdigest())
    verify_book(book)


def verify_book(book):
    """Reject mismatched fixed geometry before adopting a voyage checkpoint."""
    for leg in book['legs']:
        for name in ('fixed_approach','fixed_departure'):
            value=leg.get(name)
            if value is None:continue
            first=value.get('first_waypoint');last=value.get('last_waypoint',len(leg['coordinates'])-1)
            if (value.get('policy')!=POLICY or value.get('verified') is not True
                    or not isinstance(first,int) or not isinstance(last,int)
                    or not 0<=first<last<len(leg['coordinates'])):
                raise ValueError('invalid_fixed_port_approach')
            points=leg['coordinates'][first:last+1]
            if value.get('coordinates_sha256')!=coordinates_hash(points):
                raise ValueError('fixed_port_waypoints_checksum_changed')
            clearance=[value.get('buffer_m'),value.get('required_depth_m')]
            if (any(not isinstance(v,(int,float)) or not math.isfinite(v) for v in clearance)
                    or clearance[0]<BUFFER_M or clearance[1]<DEPTH_M):
                raise ValueError('fixed_port_clearance_changed')
            if value.get('waypoint_radius_m')!=WAYPOINT_RADIUS_M:
                raise ValueError('fixed_port_waypoint_radius_changed')


def ready(leg,index,position):
    """Join near a checked approach entry and retain harbour exit order."""
    value=certificate(leg,index)
    if value is None:return False
    if value is leg.get('fixed_departure'):return True
    first=value['first_waypoint']
    return index>first or GEO.inv(*position,*leg['coordinates'][first])[2]<1500.


def regional_plan(leg,state,policy):
    value=certificate(leg,state['waypoint']);first=value['first_waypoint'];last=value.get('last_waypoint',len(leg['coordinates'])-1)
    departing=value is leg.get('fixed_departure')
    return dict(route_key=[state['lap'],state['leg'],state['waypoint']],
        points=leg['coordinates'][first:last+1],index=state['waypoint']-first,
        mode='port_departure' if departing else 'port_approach',fixed=True,policy=policy,job=state['native']['job'],
        source=value['source'],arrival_centre=leg['coordinates'][-1],arrival_radius_m=300.,
        probe_position=leg['coordinates'][first],mapped_land_checked=True,
        regional_buffer_m=value['buffer_m'],required_depth_m=value['required_depth_m'],
        minimum_regional_clearance_m=value['minimum_depth_m'],
        scope='fixed numeric port waypoints; every native resident corridor rechecked')


def native_points(leg,index,position,maximum=1000.,limit=24):
    """Send a finite checked prefix in sailing order, retaining each bend."""
    value=certificate(leg,index)
    if value is None or not ready(leg,index,position):return []
    if value is not leg.get('fixed_departure') and index<=value['first_waypoint']:return []
    result=[];previous=position;remaining=maximum;last=value.get('last_waypoint',len(leg['coordinates'])-1)
    for point in leg['coordinates'][index:last+1]:
        heading,_,length=GEO.inv(*previous,*point)
        if length>remaining:
            result.append(list(GEO.fwd(*previous,heading,remaining)[:2]));break
        result.append(list(point));remaining-=length;previous=point
        if remaining<5. or len(result)>=limit:break
    return result


def resident_fixed_route(planner,start,points):
    """No frontier substitute for a mandatory route; unknown triangles hold it."""
    path=[list(point) for point in points]
    while len(path)>1 and math.dist(start[:2],path[0][:2])<5.:path.pop(0)
    if not path or not all(planner.segment(a,b,record=True) for a,b in zip([start,*path[:-1]],path)):
        raise RuntimeError('fixed_waypoint_corridor_pending')
    return path


def checked_route(field,origin,goal,*,step=16.,timeout=45.):
    from .navigation import DetourPlanner
    planner=DetourPlanner(field,origin,True,step=step,timeout=timeout,
                          maximum_radius=6000.,maximum_length=16000.)
    planner.radius=BUFFER_M;planner.clearance=DEPTH_M
    path=planner.route(origin,goal)
    if not all(planner.segment(a,b,record=True) for a,b in zip([origin,*path[:-1]],path)):
        raise RuntimeError('fixed_port_corridor_not_safe')
    return path,planner.minimum


def offline_planner(field,origin,*,step=16.,timeout=180.,buffer_m=BUFFER_M,depth_m=DEPTH_M):
    """Vectorised conservative node footprints; exact triangles validate every edge."""
    import numpy as np
    from scipy.ndimage import maximum_filter
    from .navigation import DetourPlanner
    from .land import LandExcludedField
    guard=field.guard if isinstance(field,LandExcludedField) else None
    base=field.field if guard else field
    terrain=base.field
    xs=origin[0]+np.arange(math.ceil((terrain.xs[0]-origin[0])/step),math.floor((terrain.xs[-1]-origin[0])/step)+1)*step
    ys=origin[1]+np.arange(math.ceil((terrain.ys[0]-origin[1])/step),math.floor((terrain.ys[-1]-origin[1])/step)+1)*step
    radius=buffer_m+step/math.sqrt(2)
    i0=np.searchsorted(terrain.xs,xs-radius,side='right')-1;i1=np.searchsorted(terrain.xs,xs+radius,side='left')
    j0=np.searchsorted(terrain.ys,ys-radius,side='right')-1;j1=np.searchsorted(terrain.ys,ys+radius,side='left')
    nx=i1-i0+1;ny=j1-j0+1
    safe=np.zeros((len(ys),len(xs)),dtype=bool)
    values=np.where(np.isfinite(terrain.heights),terrain.heights,math.inf)
    for width in np.unique(nx):
        ix=np.flatnonzero((nx==width)&(i0>=0)&(i1<len(terrain.xs)))
        for height in np.unique(ny):
            iy=np.flatnonzero((ny==height)&(j0>=0)&(j1<len(terrain.ys)))
            if not len(ix) or not len(iy):continue
            bound=maximum_filter(values,size=(int(height),int(width)),mode='constant',cval=math.inf)
            for offset in range(0,len(iy),256):
                rows=iy[offset:offset+256]
                samples=bound[np.ix_(j0[rows]+height//2,i0[ix]+width//2)]
                safe[np.ix_(rows,ix)]=samples<=-depth_m
    if guard is not None:
        import shapely
        margin=radius*math.sqrt(2)
        coverage=guard.coverage.buffer(-margin);land=guard.land.buffer(margin)
        shapely.prepare(coverage);shapely.prepare(land)
        for offset in range(0,len(ys),256):
            xx,yy=np.meshgrid(xs,ys[offset:offset+256])
            safe[offset:offset+256] &= shapely.contains_xy(coverage,xx,yy)&~shapely.intersects_xy(land,xx,yy)
    class Planner(DetourPlanner):
        def search_edge(self,a,b):
            def robust(point):
                i=round((point[0]-xs[0])/self.step);j=round((point[1]-ys[0])/self.step)
                return 0<=i<len(xs) and 0<=j<len(ys) and safe[j,i]
            if robust(a) and robust(b):return True
            key=tuple(sorted((tuple(a),tuple(b))))
            if key not in edges:edges[key]=self.segment(a,b)
            return edges[key]
        def segment(self,a,b,record=False):
            # Reject an entire land-crossing shortcut before sampling thousands
            # of numeric footprints. The approved path still uses exact terrain.
            if guard is not None and not guard.segment(a,b,self.radius+.5):return False
            # All vertices enclosing a swept rectangle bound every interpolated
            # triangle. This proof is conservative; near banks or unknown cells
            # retain the exact one-metre sweep below.
            lo=np.minimum(a[:2],b[:2])-self.radius-.5
            hi=np.maximum(a[:2],b[:2])+self.radius+.5
            if (lo[0]>=terrain.xs[0] and hi[0]<=terrain.xs[-1]
                    and lo[1]>=terrain.ys[0] and hi[1]<=terrain.ys[-1]
                    and max(math.dist(a[:2],self.origin),math.dist(b[:2],self.origin))<=self.maximum_radius):
                left=np.searchsorted(terrain.xs,lo[0],side='right')-1;right=np.searchsorted(terrain.xs,hi[0],side='left')
                bottom=np.searchsorted(terrain.ys,lo[1],side='right')-1;top=np.searchsorted(terrain.ys,hi[1],side='left')
                floor=values[bottom:top+1,left:right+1].max();depth=min(a[2],b[2],0.)-floor
                if floor<0 and depth>=self.clearance:
                    if record:self.minimum=min(self.minimum,float(depth))
                    return True
            return super().segment(a,b,record)
        def safe(self,xy,z,radius=None):
            if radius is not None and abs(radius-(self.radius+self.step/math.sqrt(2)))<1e-8:
                i=round((xy[0]-xs[0])/self.step);j=round((xy[1]-ys[0])/self.step)
                if not (0<=i<len(xs) and 0<=j<len(ys)):return False
                if safe[j,i] and z>=0:return True
                # Rectangular upper bounds reject diagonal narrow channels.
                # Check that node's exact swept footprint before excluding it.
                return super().safe(xy,z,self.radius)
            return super().safe(xy,z,radius)
    edges={}
    planner=Planner(field,origin,True,step=step,timeout=timeout,maximum_radius=160000.,maximum_length=160000.)
    planner.radius=buffer_m;planner.clearance=depth_m
    return planner


def simulate(points,fields,*,heading_offset=0.,lateral_offset=0.,dt=.5,arrival_radius_m=300.):
    """Bounded speed/yaw response over real terrain; separate from Gazebo physics.

    Preserve every waypoint. Check actual swept hull motion, tracking error and
    measured arrival eligibility; no wall-time or geometric teleport advances it.
    """
    import numpy as np
    from .navigation import DetourPlanner
    from .port_approach import manoeuvre_speed
    xy=np.asarray(points,dtype=float)
    direction=xy[1]-xy[0];direction=direction/np.linalg.norm(direction)
    position=xy[0]+np.array([-direction[1],direction[0]])*lateral_offset
    heading=math.atan2(direction[1],direction[0])+heading_offset
    speed=0.;index=1;elapsed=0.;maximum_error=0.;minimum=math.inf;trace=[position.tolist()]
    length=sum(math.dist(a,b) for a,b in zip(xy,xy[1:]))
    deadline=max(600.,length/.6+300.)
    validators=[]
    for field in fields:
        validator=DetourPlanner(field,position.tolist()+[0.],True,maximum_radius=100000.,maximum_length=100000.)
        validator.radius=3.5
        validators.append(validator)
    while elapsed<deadline:
        while index<len(xy)-1 and math.dist(position,xy[index])<WAYPOINT_RADIUS_M:index+=1
        if index==len(xy)-1 and math.dist(position,xy[-1])<=arrival_radius_m:
            return dict(passed=True,elapsed_sim_seconds=elapsed,waypoints_reached=index,
                        arrival_distance_m=math.dist(position,xy[-1]),maximum_tracking_error_m=maximum_error,
                        minimum_hull_depth_m=minimum,trace=trace)
        goal=xy[index];delta=goal-position;desired=math.atan2(delta[1],delta[0])
        error=(desired-heading+math.pi)%(2*math.pi)-math.pi
        request=manoeuvre_speed(4.,position.tolist(),xy.tolist(),index,True,math.dist(position,xy[-1]))
        request=min(request,max(.5,4.*max(0.,math.cos(error))))
        speed+=max(-.65*dt,min(.35*dt,request-speed))
        heading+=max(-.30*dt,min(.30*dt,1.4*error*dt))
        next_position=position+speed*dt*np.array([math.cos(heading),math.sin(heading)])
        # A bounded lateral wave velocity tests tracking and clearance.
        next_position+=.08*math.sin(elapsed*.7)*dt*np.array([-math.sin(heading),math.cos(heading)])
        previous=xy[index-1];axis=goal-previous
        fraction=max(0.,min(1.,float(np.dot(next_position-previous,axis)/np.dot(axis,axis))))
        tracking=math.dist(next_position,previous+axis*fraction)
        maximum_error=max(maximum_error,tracking)
        if tracking>BUFFER_M-3.5:
            raise RuntimeError('fixed_port_tracking_margin_exceeded:'+str(round(tracking,3)))
        checked=False
        for validator in validators:
            if validator.segment([*position,0.],[*next_position,0.],record=True):
                minimum=min(minimum,validator.minimum);checked=True;break
        if not checked:raise RuntimeError('fixed_port_simulated_hull_clearance_failed')
        position=next_position;elapsed+=dt
        if len(trace)<2 or math.dist(position,trace[-1])>=10.:trace.append(position.tolist())
    raise RuntimeError('fixed_port_simulation_stalled')
