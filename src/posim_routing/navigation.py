"""Bounded terrain-only A* routing on immutable, confirmed collision triangles.

The horizontal disk footprint is conservatively enclosed by a square. Height
maxima are computed by clipping actual mesh triangles, not by extrapolation or
by sampling just a centreline. Unknown cells are obstacles.
"""
import heapq
import math
import time
import numpy as np


class TerrainField:
    def __init__(self, xs, ys, heights):
        self.xs, self.ys, self.heights = map(np.asarray, (xs, ys, heights))
        if len(xs)<2 or len(ys)<2 or self.heights.shape!=(len(ys),len(xs)):
            raise RuntimeError('invalid_navigation_terrain')

    @classmethod
    def tiles(cls, tiles):
        if not tiles:raise RuntimeError('route_outside_confirmed_terrain')
        xs=np.unique(np.concatenate([t['xs'] for t in tiles]));ys=np.unique(np.concatenate([t['ys'] for t in tiles]))
        # Preserve missing tile gaps as NaN, including cells bridging the gap.
        dx=min(np.diff(xs));dy=min(np.diff(ys))
        xs=np.arange(round((xs[-1]-xs[0])/dx)+1)*dx+xs[0]
        ys=np.arange(round((ys[-1]-ys[0])/dy)+1)*dy+ys[0]
        heights=np.full((len(ys),len(xs)),np.nan)
        for t in tiles:
            ix=np.rint((np.asarray(t['xs'])-xs[0])/dx).astype(int)
            iy=np.rint((np.asarray(t['ys'])-ys[0])/dy).astype(int)
            heights[np.ix_(iy,ix)]=t['heights']
        return cls(xs,ys,heights)

    @staticmethod
    def clip(poly, axis, boundary, sign):
        out=[]
        previous=poly[-1];pd=sign*(previous[axis]-boundary)
        for current in poly:
            cd=sign*(current[axis]-boundary)
            if (cd>=0)!=(pd>=0):
                f=pd/(pd-cd)
                out.append(tuple(a+(b-a)*f for a,b in zip(previous,current)))
            if cd>=0:out.append(current)
            previous,pd=current,cd
        return out

    def maximum(self, x, y, radius):
        """Exact maximum of the piecewise planar floor over an XY square."""
        lo_x,hi_x,lo_y,hi_y=x-radius,x+radius,y-radius,y+radius
        xs,ys,h=self.xs,self.ys,self.heights
        if lo_x<xs[0] or hi_x>xs[-1] or lo_y<ys[0] or hi_y>ys[-1]:return None
        i0=max(0,int(np.searchsorted(xs,lo_x,side='right'))-1)
        i1=min(len(xs)-2,int(np.searchsorted(xs,hi_x,side='left'))-1)
        j0=max(0,int(np.searchsorted(ys,lo_y,side='right'))-1)
        j1=min(len(ys)-2,int(np.searchsorted(ys,hi_y,side='left'))-1)
        maximum=-math.inf
        for j in range(j0,j1+1):
            for i in range(i0,i1+1):
                a=(xs[i],ys[j],h[j,i]);b=(xs[i+1],ys[j],h[j,i+1])
                c=(xs[i],ys[j+1],h[j+1,i]);d=(xs[i+1],ys[j+1],h[j+1,i+1])
                if not all(math.isfinite(v[2]) for v in (a,b,c,d)):return None
                for poly in ([a,b,d],[a,d,c]):
                    for axis,boundary,sign in ((0,lo_x,1),(0,hi_x,-1),(1,lo_y,1),(1,hi_y,-1)):
                        if not poly:break
                        poly=self.clip(poly,axis,boundary,sign)
                    if poly:maximum=max(maximum,max(v[2] for v in poly))
        return maximum if math.isfinite(maximum) else None


class DetourPlanner:
    def __init__(self, field, start, boat, step=4., timeout=25., maximum_radius=600., maximum_length=2000.):
        self.field,self.origin,self.step=field,start[:2],step
        self.maximum_radius,self.maximum_length=maximum_radius,maximum_length
        self.clearance=1.5 if boat else 1.
        self.hull_radius=3.5 if boat else .5
        # The 6 m-class surface hull has appreciable wave-driven sway while
        # steering. Reserve this in planning as well as the runtime guard;
        # accepting a wider trace without expanding terrain clearance is unsafe.
        self.tracking_margin=4. if boat else 2.
        self.radius=self.hull_radius+self.tracking_margin
        self.deadline=time.monotonic()+timeout
        self.minimum=math.inf
        self.expanded=0

    def safe(self, xy, z, radius=None):
        if math.dist(xy,self.origin)>self.maximum_radius:return False
        floor=self.field.maximum(*xy,self.radius if radius is None else radius)
        return floor is not None and floor<0 and z-floor>=self.clearance

    def segment(self, a, b, record=False):
        distance=math.dist(a[:2],b[:2]);count=max(1,math.ceil(distance/1.))
        # Squares enlarged by half the sample separation cover the entire swept
        # disk between evaluations. The deeper endpoint bounds sloping Z legs.
        radius=self.radius+distance/count/2.;z=min(a[2],b[2],0.)
        minimum=math.inf
        for k in range(count+1):
            f=k/count;xy=[a[i]+(b[i]-a[i])*f for i in (0,1)]
            if math.dist(xy,self.origin)>self.maximum_radius:return False
            floor=self.field.maximum(*xy,radius)
            if floor is None or floor>=0 or z-floor<self.clearance:return False
            minimum=min(minimum,z-floor)
        if record:self.minimum=min(self.minimum,minimum)
        return True

    def route(self, start, goal):
        z=min(start[2],goal[2],0.)
        if not self.safe(start[:2],start[2]):raise RuntimeError('route_start_unsafe')
        if not self.safe(goal[:2],goal[2]):raise RuntimeError('route_goal_unsafe')
        if self.segment(start,goal):return [goal]
        # Use the conservative deepest leg depth for search and every shortcut.
        if not self.safe(start[:2],z) or not self.safe(goal[:2],z):
            raise RuntimeError('route_depth_transition_unsafe')
        ox,oy=start[:2];step=self.step;cache={};cost={(0,0):0.};parent={}
        def position(node):return [ox+node[0]*step,oy+node[1]*step,z]
        def free(node):
            if node not in cache:cache[node]=self.safe(position(node)[:2],z,self.radius+step/math.sqrt(2))
            return cache[node]
        heap=[(math.dist(start[:2],goal[:2]),0.,(0,0))];closed=set();found=None
        while heap:
            if time.monotonic()>self.deadline:raise RuntimeError('route_planning_budget_exceeded')
            _,g,node=heapq.heappop(heap)
            if node in closed:continue
            closed.add(node);self.expanded+=1;p=position(node)
            if math.dist(p[:2],goal[:2])<=step*4 and self.segment(p,goal):found=node;break
            for dx,dy in ((1,0),(-1,0),(0,1),(0,-1),(1,1),(1,-1),(-1,1),(-1,-1)):
                nxt=(node[0]+dx,node[1]+dy)
                if nxt in closed or not free(nxt):continue
                if dx and dy and (not free((node[0]+dx,node[1])) or not free((node[0],node[1]+dy))):continue
                if node==(0,0) and not self.segment(start,position(nxt)):continue
                ng=g+step*math.hypot(dx,dy)
                if ng>=cost.get(nxt,math.inf):continue
                cost[nxt]=ng;parent[nxt]=node
                heapq.heappush(heap,(ng+math.dist(position(nxt)[:2],goal[:2]),ng,nxt))
        if found is None:raise RuntimeError('route_no_safe_detour')
        nodes=[];n=found
        while n!=(0,0):nodes.append(position(n));n=parent[n]
        path=[[*start[:2],z]]+list(reversed(nodes))+[[*goal[:2],z]]
        simplified=[path[0]];i=0
        while i<len(path)-1:
            if time.monotonic()>self.deadline:raise RuntimeError('route_planning_budget_exceeded')
            next_index=i+1
            for j in range(len(path)-1,i,-1):
                if self.segment(path[i],path[j]):next_index=j;break
            simplified.append(path[next_index]);i=next_index
        lengths=[math.dist(a[:2],b[:2]) for a,b in zip(simplified,simplified[1:])]
        total=sum(lengths);travel=0.;result=[]
        for p,length in zip(simplified[1:],lengths):
            travel+=length
            result.append([*p[:2],start[2]+(goal[2]-start[2])*(travel/total)])
        result[-1]=goal
        return result

    def plan(self, start, points):
        result=[];previous=start;detoured=0
        for point in points:
            route=self.route(previous,point['position'])
            if len(route)>1:detoured+=1
            for position in route[:-1]:
                result.append(dict(phase='transit',position=position,hold_seconds=0.,
                                   destination_number=point.get('number')))
            result.append(point)
            previous=point['position']
        if len(result)>160:raise RuntimeError('route_transit_limit_160')
        previous=start
        for p in result:
            if not self.segment(previous,p['position'],record=True):raise RuntimeError('route_validation_failed')
            previous=p['position']
        length=sum(math.dist(a,b['position']) for a,b in zip([start]+[p['position'] for p in result[:-1]],result))
        if length>self.maximum_length:raise RuntimeError('route_limit_2000_m')
        return result,dict(algorithm='buffered-terrain-a-star',detoured_legs=detoured,
            transit_points=sum(p['phase']=='transit' for p in result),grid_step_m=self.step,maximum_radius_m=self.maximum_radius,maximum_length_m=self.maximum_length,
            hull_radius_m=self.hull_radius,tracking_margin_m=self.tracking_margin,
            search_nodes=self.expanded,minimum_clearance_m=self.minimum,route_length_m=length,
            scope='confirmed static terrain only; no AIS or unrepresented obstacles')
