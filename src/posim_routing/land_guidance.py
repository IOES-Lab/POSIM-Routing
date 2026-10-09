"""Bounded mapped-water guidance; numeric depth must approve every local leg."""
import heapq
import math
import time

import numpy as np


def detour(guard, start, target, step=150., radius=40., timeout=12.):
    """Return a complete mapped-water route, including necessary initial retreat.

    This exclusion search neither creates water nor approves bathymetry. Its
    endpoints and shortcuts enclose the same swept square as the local planner.
    """
    deadline=time.monotonic()+timeout
    if not guard.free(*start,radius) or not guard.free(*target,radius):
        raise RuntimeError('mapped_guidance_endpoint_blocked')
    if guard.segment(start,target,radius):return [list(target)]
    pad=24000.
    lo=np.minimum(start,target)-pad;hi=np.maximum(start,target)+pad
    if np.max(hi-lo)>128000.:
        raise RuntimeError('mapped_guidance_horizon_exceeded')
    xs=np.arange(lo[0],hi[0]+step/2,step);ys=np.arange(lo[1],hi[1]+step/2,step)
    safe=guard.grid(xs,ys,radius+step/math.sqrt(2))
    def xy(node):return [float(xs[node[1]]),float(ys[node[0]])]
    def anchor(point):
        j=int(np.argmin(abs(ys-point[1])));i=int(np.argmin(abs(xs-point[0])))
        candidates=[(jj,ii) for jj in range(max(0,j-2),min(len(ys),j+3))
                    for ii in range(max(0,i-2),min(len(xs),i+3))]
        candidates.sort(key=lambda n:math.dist(xy(n),point))
        for node in candidates:
            if safe[node] and guard.segment(point,xy(node),radius):return node
        raise RuntimeError('mapped_guidance_anchor_blocked')
    origin,goal=anchor(start),anchor(target)
    costs={origin:0.};parents={};heap=[(math.dist(xy(origin),xy(goal)),0.,origin)]
    closed=set();found=False
    offsets=((0,1),(0,-1),(1,0),(-1,0),(1,1),(1,-1),(-1,1),(-1,-1))
    while heap:
        if time.monotonic()>deadline:raise RuntimeError('mapped_guidance_budget_exceeded')
        _,cost,node=heapq.heappop(heap)
        if node in closed:continue
        closed.add(node)
        if node==goal:found=True;break
        for j,i in offsets:
            nxt=(node[0]+j,node[1]+i)
            if not (0<=nxt[0]<len(ys) and 0<=nxt[1]<len(xs)) or not safe[nxt] or nxt in closed:continue
            if i and j and (not safe[node[0]+j,node[1]] or not safe[node[0],node[1]+i]):continue
            value=cost+step*math.hypot(i,j)
            if value>=costs.get(nxt,math.inf):continue
            costs[nxt],parents[nxt]=value,node
            heapq.heappush(heap,(value+math.dist(xy(nxt),xy(goal)),value,nxt))
    if not found:raise RuntimeError('mapped_guidance_no_water_detour')
    nodes=[goal]
    while nodes[-1]!=origin:nodes.append(parents[nodes[-1]])
    path=[list(start),*[xy(n) for n in reversed(nodes)],list(target)]
    # Greedy checked shortcuts preserve the complete around-land connection.
    result=[];index=0
    while index<len(path)-1:
        if time.monotonic()>deadline:raise RuntimeError('mapped_guidance_budget_exceeded')
        chosen=index+1
        for end in range(len(path)-1,index,-1):
            if guard.segment(path[index],path[end],radius):chosen=end;break
        result.append(path[chosen]);index=chosen
    if not all(guard.segment(a,b,radius) for a,b in zip([start,*result[:-1]],result)):
        raise RuntimeError('mapped_guidance_validation_failed')
    if sum(math.dist(a,b) for a,b in zip([start,*result[:-1]],result))>160000.:
        raise RuntimeError('mapped_guidance_length_exceeded')
    return result


def local_target(start, points, horizon=6500.):
    """Take a checked prefix, preserving each bend before the numeric horizon."""
    point=points[0];length=math.dist(start,point)
    if length<=horizon:return list(point)
    return [start[i]+(point[i]-start[i])*horizon/length for i in (0,1)]
