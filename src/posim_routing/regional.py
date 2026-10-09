"""Background regional lookahead for coarse maritime graph coast crossings.

Only plans geographic waypoints. Movement and local swept-hull approval remain
in native Gazebo. Downloading/planning never pauses the promotional world.
"""
import asyncio,math,time
from pyproj import Geod
GEO=Geod(ellps='WGS84')
REGIONAL_POLICY='buffered-water-v8-mapped-land-exclusion'
REGIONAL_BUFFER_M=40.
REGIONAL_DEPTH_M=2.

def distance(a,b):return GEO.inv(*a,*b)[2]

def corridor(terrain,spec,start,target,guard=None):
    from posim_terrain.terrain import projection
    from .navigation import TerrainField,DetourPlanner
    from .planning import CachedTerrain
    _,forward,inverse=projection(spec)
    from posim_terrain.canal_override import in_region
    canal=in_region(start)
    field=CachedTerrain(TerrainField(terrain['xs'],terrain['ys'],terrain['heights']))
    if guard is not None:
        from .land import LandExcludedField
        field=LandExcludedField(field,guard)
    origin=[*forward.transform(*start),0.]
    heading,_,remaining=GEO.inv(*start,*target)
    deadline=time.monotonic()+60;errors=[]
    for offset in (0,15,-15,30,-30,60,-60,90,-90,120,-120,150,-150,180):
        lon,lat,_=GEO.fwd(*start,heading+offset,min(750. if canal else 7500.,remaining))
        goal=[*forward.transform(lon,lat),0.]
        planner=DetourPlanner(field,origin,True,step=16 if canal else 80,timeout=min(25,max(.1,deadline-time.monotonic())),maximum_radius=1800 if canal else 9500,maximum_length=5000 if canal else 40000)
        # Regional pixels are coarser than the resident collision mesh. A
        # barely navigable pixel led into a channel closed by the detailed
        # mesh, so the local frontier search circled an unreachable goal.
        # Reserve the same depth and coastal footprint as coastal lookahead,
        # along the entire path, not just at its endpoint.
        planner.radius=12. if canal else REGIONAL_BUFFER_M;planner.clearance=REGIONAL_DEPTH_M
        if not planner.safe(goal[:2],0.):continue
        try:
            path=planner.route(origin,goal)
            if sum(math.dist(a[:2],b[:2]) for a,b in zip([origin]+path[:-1],path))>40000:continue
            if not all(planner.segment(a,b,record=True) for a,b in zip([origin]+path[:-1],path)):continue
            points=[list(inverse.transform(*p[:2])) for p in path]
            return dict(points=points,index=0,planning_seconds=60-(deadline-time.monotonic()),
                        search_nodes=planner.expanded,minimum_regional_clearance_m=planner.minimum,
                        regional_buffer_m=planner.radius,required_depth_m=planner.clearance,
                        scope='numeric regional lookahead; every native local corridor is checked separately')
        except RuntimeError as e:errors.append(str(e))
        if time.monotonic()>deadline:break
    raise RuntimeError('regional_corridor_unavailable:'+','.join(errors[-2:]))

class RegionalNavigator:
    def __init__(self,rpc,coastal=False,before_prepare=None):
        self.before_prepare=before_prepare
        self.rpc=rpc;self.task=None;self.key=None;self.retry=0
        self.coastal=coastal;self.coast_index=None;self.index_task=None;self.probed=None
        self.progress=None
    def discard(self,state,reason):
        state.pop('regional_navigation',None)
        state['regional_planning']=dict(status='expired',reason=reason)
        (state.get('native',{}).get('navigation') or {}).pop('route_key',None)
        self.retry=0;self.progress=None
    def active_point(self,state,key):
        plan=state.get('regional_navigation') or {}
        if plan.get('route_key')!=key or not plan.get('points'):return None
        from .port_approach import ARRIVAL_RADIUS_M
        if plan.get('mode')=='port_approach' and plan.get('arrival_radius_m')!=ARRIVAL_RADIUS_M:
            self.discard(state,'port_arrival_region_changed')
            return None
        if plan.get('policy')!=REGIONAL_POLICY:
            self.discard(state,'regional_clearance_policy_changed')
            return None
        from .course import passed_vertex
        while plan['index']<len(plan['points']):
            index=plan['index'];point=plan['points'][index]
            reached=distance(state['position'],point)<(12 if plan.get('mode')=='port_approach' else 80)
            previous=plan['points'][index-1] if index else plan.get('probe_position')
            passed=(plan.get('mode') in ('coastal','coastal_approach') and previous is not None
                    and index<len(plan['points'])-1 and passed_vertex(previous,point,plan['points'][index+1],
                        state['position'],maximum_offset=200.,minimum_leg=0.,corner_margin=40.))
            if not reached and not passed:break
            plan['index']+=1
        if plan['index']>=len(plan['points']):
            # Only consume an explicitly validated finite onward route. This
            # supplies runway during renewal without extrapolating a tangent.
            if plan.get('onward_verified') and plan.get('onward_points'):
                plan['points'].extend(plan.pop('onward_points'));plan['onward_used']=True
            else:return None
        point=plan['points'][plan['index']]
        # A local detour may bypass the coarse frontier without entering its
        # 80 m arrival circle. Reusing that persisted goal hundreds of km
        # later steered the voyage backwards. Only use regional goals within
        # their planning horizon; the native collision-checked pilot still
        # approves every replacement corridor.
        if distance(state['position'],point)>12000:
            self.discard(state,'outside_regional_horizon')
            return None
        # Local frontier endpoints may all be reached while the regional goal
        # never gets closer. Observe that goal across replans and retreats;
        # changing the local pilot corridor must not reset this watchdog.
        native=state.get('native') or {};sim_time=native.get('sim_time')
        if sim_time is not None and math.isfinite(sim_time):
            remaining=distance(state['position'],point)
            progress_key=(native.get('nonce'),tuple(key),plan['index'],tuple(point))
            if self.progress is None or self.progress[0]!=progress_key or sim_time<self.progress[1] or remaining<self.progress[2]-50:
                self.progress=(progress_key,sim_time,remaining)
            elif sim_time-self.progress[1]>max(300.,self.progress[2]/max(1.5,state.get('commanded_speed_mps',0))*3):
                self.discard(state,'regional_goal_progress_stalled')
                return None
        return point
    def update(self,state,target,needed=False,terminal=False,retain_coast=False):
        from posim_terrain.canal_override import in_region
        canal=in_region(state['position'])
        key=[state['lap'],state['leg'],state['waypoint']]
        from .port_approach import APPROACH_DISTANCE_M
        terminal=terminal and distance(state['position'],target)<APPROACH_DISTANCE_M
        state.pop('regional_waiting',None)
        if self.key!=key:
            if self.task:self.task.cancel()
            self.task=None;self.key=key;self.retry=0
            self.probed=None
            plan=state.get('regional_navigation') or {};old=plan.get('route_key')
            if (retain_coast and not terminal and not canal and old and len(old)==3
                    and old[:2]==key[:2] and old[2]<key[2]
                    and plan.get('policy')==REGIONAL_POLICY
                    and plan.get('mode') in ('coastal','coastal_approach')):
                plan['route_key']=list(key);plan.pop('continuation',None)
                state['regional_handoff']=dict(previous_key=old,route_key=list(key),
                    position=list(state['position']),reason='measured_coastal_shape_advance')
                self.progress=None
        nearest=None
        if self.coastal and not canal:
            if self.index_task is None and self.coast_index is None:
                from .coastal import CoastIndex
                self.index_task=asyncio.create_task(asyncio.to_thread(CoastIndex))
            if self.index_task and self.index_task.done():
                try:self.coast_index=self.index_task.result()
                except Exception as e:
                    # A discovery failure leaves terrain-only routing intact.
                    state['coastal_navigation']=dict(status='discovery_unavailable',reason=str(e)[:160])
                    self.coastal=False
                self.index_task=None
            if self.coast_index:
                from .coastal import COAST_DISCOVERY_M
                separation,shore=self.coast_index.nearest(state['position'])
                if separation<COAST_DISCOVERY_M:
                    nearest=(separation,shore)
                    if separation>2500 and hasattr(self.coast_index,'progressive_shore'):
                        nearest=self.coast_index.progressive_shore(state['position'],target)
                state['coastal_navigation']=dict(status='coastal_region' if nearest else 'open_ocean',
                                                discovery_distance_m=separation)
        if canal and (state.get('regional_navigation') or {}).get('mode') not in (None,'canal'):
            self.discard(state,'suez_fine_grid')
        point=self.active_point(state,key)
        plan=state.get('regional_navigation') or {}
        # A mapped around-land guide must renew from its checked endpoint. It
        # cannot fall back to the original straight line across that land.
        guide_exhausted=(plan.get('route_key')==key and plan.get('mapped_guidance')
                         and plan.get('points') and plan['index']>=len(plan['points']))
        needed=needed or bool(guide_exhausted)
        approach_runway=(plan.get('mode')=='port_approach_lookahead' and point is not None
                         and distance(state['position'],plan['probe_position'])<500.)
        prefer_port=terminal and plan.get('mode')!='port_approach' and not approach_runway
        if terminal and point is not None and not prefer_port and not (self.task and self.task.done()):return point
        # Prepare the next coastal lookahead while the current checked plan
        # still has runway. Otherwise its end briefly falls back to an offshore
        # graph vertex and the ship turns out, then back when download finishes.
        renew_coast=(point is not None and nearest is not None and plan.get('mode') in ('coastal','coastal_approach')
                     and distance(state['position'],plan['points'][-1])<2500
                     and distance(state['position'],plan.get('probe_position',state['position']))>=1000
                     and (self.probed is None or distance(self.probed,state['position'])>=1000))
        from .coastal import WAYPOINT_FORMAT
        renew_coast=renew_coast or (point is not None and nearest is not None
                                    and plan.get('mode')=='coastal'
                                    and (not plan.get('onward_verified') or plan.get('waypoint_format')!=WAYPOINT_FORMAT))
        coast_needed=nearest is not None and (self.probed is None or distance(self.probed,state['position'])>=3000)
        prefer_coast=(point is not None and coast_needed and not needed and
                      (state.get('regional_navigation') or {}).get('mode') not in ('coastal','coastal_approach'))
        if point is not None and not prefer_port and not prefer_coast and not renew_coast and not (self.task and self.task.done()):return point
        # A clear local segment is not proof that its regional destination is
        # reachable. Rebuild an expired/stalled goal even when the most recent
        # local segment did not itself need a detour or coast discovery.
        expired=state.get('regional_planning') or {}
        needed=needed or (expired.get('status')=='expired' and expired.get('reason') in
                         ('regional_clearance_policy_changed','regional_goal_progress_stalled','suez_fine_grid'))
        if self.task and self.task.done():
            try:
                value=self.task.result()
                # A failed optional preference must preserve a previously
                # checked detour while waiting for a better coastal region.
                coast_plan=(plan.get('route_key')==key and plan.get('mode') in ('coastal','coastal_approach')
                            and plan.get('policy')==REGIONAL_POLICY)
                preserve_empty=(not value['points'] and (point is not None or
                                (coast_plan and nearest is not None and not terminal and not needed)))
                if not preserve_empty:state['regional_navigation']=value
                self.probed=list(value['probe_position']) if value.get('mode') not in ('coastal','coastal_approach') else None
                state['regional_planning']=dict(status='ready' if value['points'] else 'course_retained',
                                                job=value['job'],mode=value.get('mode','detour'),coastal_preference_error=value.get('coastal_preference_error'))
                if preserve_empty and coast_plan:
                    self.probed=None;self.retry=time.monotonic()+60
                    state['regional_planning']['status']='retrying'
            except (Exception,asyncio.CancelledError) as e:
                state['regional_planning']=dict(status='retrying',reason=str(e)[:160]);self.retry=time.monotonic()+60
            self.task=None
            point=self.active_point(state,key)
            if point is not None:
                if state['regional_navigation'].get('mode') in ('coastal','coastal_approach','port_approach'):
                    # Apply a completed background preference once, instead
                    # of first finishing an old corridor away from the coast.
                    (state.get('native',{}).get('navigation') or {}).pop('route_key',None)
                state['regional_planning'].update(status='ready',active_job=state['regional_navigation']['job']);return point
        coast_needed=nearest is not None and (self.probed is None or distance(self.probed,state['position'])>=3000)
        if (prefer_port or needed or coast_needed or renew_coast) and not self.task and time.monotonic()>=self.retry:
            state['regional_planning']=dict(status='background',mode='port_approach' if terminal else 'coastal' if coast_needed else 'detour')
            self.task=asyncio.create_task(self.prepare_port(state,target,key) if terminal else self.prepare(state,target,key,nearest,needed))
        if point is not None:return point
        plan=state.get('regional_navigation') or {}
        if (plan.get('route_key')==key and plan.get('policy')==REGIONAL_POLICY
                and (plan.get('mode') in ('coastal','coastal_approach') or plan.get('mapped_guidance')) and plan.get('points')
                and plan['index']>=len(plan['points'])
                and distance(state['position'],plan['points'][-1])<1500
                and (nearest is not None or self.task or self.index_task or plan.get('mapped_guidance'))):
            # An exhausted checked path is not permission to steer offshore.
            # The caller holds thrust, keeping physics and sensors running.
            # Retry backoff and optional empty results retain this boundary.
            plan.pop('continuation',None)
            state['regional_waiting']=dict(reason='coastal_renewal',job=plan['job'],
                                           endpoint=plan['points'][-1])
            return plan['points'][-1]
        if (nearest is not None or guide_exhausted or needed) and (self.task or time.monotonic()<self.retry):
            state['regional_waiting']=dict(reason='land_checked_coastal_route_pending')
            return state['position']
        return target
    async def prepare_port(self,state,target,key):
        # A port's narrow entrance is resolved by the same triangles already
        # in Gazebo. A 20 km / 257 grid can close that entrance and create a
        # cyclic offshore frontier. No scenery preference replaces this goal.
        from .port_approach import corridor, ArrivalOutsideResidentTerrain
        native=await self.rpc('/promotional/state')
        stream=await self.rpc('/stream/state')
        epoch=(native.get('origin') or {}).get('epoch',0)
        if stream.get('nonce')!=native['session_nonce']:raise RuntimeError('port_world_changed')
        tiles=await asyncio.gather(*(self.rpc('/stream/tiles/'+t['key']) for t in stream['tiles']))
        spec=(native.get('origin') or {}).get('spec')
        if not spec:
            job=await self.rpc('/terrain/jobs/'+state['native']['job']);spec=job['spec']
        geo=native['geographic'];current=[geo['longitude'],geo['latitude']]
        from .land import guard_from_spec
        from posim_terrain.canal_override import in_region
        guard=None if in_region(current) else await asyncio.to_thread(guard_from_spec,spec,64000.)
        try:
            value=await asyncio.to_thread(corridor,tiles,spec,current,target,guard=guard)
        except ArrivalOutsideResidentTerrain:
            # Robot-centred streaming cannot load a distant entrance while the
            # ship is stationary. Advance on verified regional water, with the
            # native resident corridor rechecking every movement command.
            value=await self.prepare(state,target,key,needed=True,extent=6000)
            value.update(mode='port_approach_lookahead',arrival_centre=list(target))
            return value
        latest=await self.rpc('/promotional/state')
        if latest['session_nonce']!=native['session_nonce'] or (latest.get('origin') or {}).get('epoch',0)!=epoch:
            raise RuntimeError('port_frame_changed')
        value.update(route_key=key,job=state['native']['job'],source='resident collision terrain',
                     policy=REGIONAL_POLICY,probe_position=current,mapped_land_checked=guard is not None)
        return value
    async def terrain_region(self, centre, extent):
        """Try a smaller verified region when the baseline provider is offline."""
        for width in (extent, 6000) if extent > 6000 else (extent,):
            job=await self.rpc('/terrain/jobs',dict(latitude=centre[1],longitude=centre[0],
                width_m=width,height_m=width,samples=257,source_policy='auto'),20)
            end=time.monotonic()+620
            while job['status'] in ('queued','running') and time.monotonic()<end:
                await asyncio.sleep(2);job=await self.rpc('/terrain/jobs/'+job['id'])
            if job['status']=='ready':return job
            if width==6000 or job['status']!='failed':break
            try:
                sources=await self.rpc('/terrain/jobs/'+job['id']+'/files/source-providers.json')
            except Exception:break
            if (sources.get('baseline_status') or {}).get('status')!='unavailable':break
        raise RuntimeError('regional_terrain_unavailable')
    async def prepare(self,state,target,key,nearest=None,needed=True,extent=None):
        from posim_terrain.canal_override import in_region
        canal=in_region(state['position'])
        if self.coastal and not canal and nearest is None and self.index_task is not None:
            # Policy migration can request a replan while the discovery index
            # is still warming. Do not commit another offshore detour first.
            try:
                index=await asyncio.shield(self.index_task)
                from .coastal import COAST_DISCOVERY_M
                separation,shore=index.nearest(state['position'])
                if separation<COAST_DISCOVERY_M:
                    nearest=index.progressive_shore(state['position'],target) if separation>2500 else (separation,shore)
            except Exception:pass  # Numeric collision routing remains available.
        pos=list(state['position'])
        centre=pos
        if nearest and 6000<nearest[0]<15000:
            # Keep the vessel inside a 20 km region while including a more
            # distant coast. The 1:10M discovery line never supplies depth.
            heading,_,_=GEO.inv(*pos,*nearest[1])
            centre=list(GEO.fwd(*pos,heading,min(6000.,nearest[0]-6000.))[:2])
        extent=extent or (4000 if canal else 20000)
        if self.before_prepare is not None:await self.before_prepare()
        job=await self.terrain_region(centre,extent)
        grid=await self.rpc('/terrain/jobs/'+job['id']+'/files/terrain.json',timeout=20)
        guard=None
        if not canal:
            from .land import guard_from_spec
            guard=await asyncio.to_thread(guard_from_spec,job['spec'],64000.)
        current=list(state['position'])
        course_target=target;guidance=None
        if guard is not None:
            from posim_terrain.terrain import projection
            from .land_guidance import detour,local_target
            _,forward,inverse=projection(job['spec'])
            origin=forward.transform(*current);destination=forward.transform(*target)
            # Check a complete around-land connection before selecting a local
            # numeric goal. A peninsula can require temporary distance loss.
            length=math.dist(origin,destination)
            if length>50000.:
                destination=[origin[i]+(destination[i]-origin[i])*50000./length for i in (0,1)]
            points=await asyncio.to_thread(detour,guard,origin,destination)
            if len(points)>1:
                target=list(inverse.transform(*local_target(origin,points)))
                guidance=dict(points=[list(inverse.transform(*p)) for p in points],
                              course_target=list(course_target),
                              scope='mapped land exclusion only; numeric depth checked locally')
        value=None;coast_error=None
        if not canal and (nearest or needed):
            from .coastal import coastal_corridor
            try:value=await asyncio.to_thread(coastal_corridor,grid,job['spec'],current,target,guard=guard)
            except RuntimeError as error:coast_error=str(error)[:160]
        if value is None:
            current=list(state['position'])
            approach=None
            if nearest:
                from .coastal import coastal_approach
                approach=coastal_approach(current,target,nearest[1])
            if approach:
                try:
                    # This small step is checked against real numeric depth,
                    # then checked again against the live Gazebo collision mesh.
                    value=await asyncio.to_thread(corridor,grid,job['spec'],current,approach,guard=guard)
                    endpoint=value['points'][-1]
                    if (distance(current,target)-distance(endpoint,target)<500 or
                        distance(current,nearest[1])-distance(endpoint,nearest[1])<500):
                        value=None
                    else:
                        from .coastal import STANDOFF_M
                        value.update(mode='coastal_approach',preferred_standoff_m=STANDOFF_M)
                except RuntimeError as error:
                    coast_error=str(error)[:160]
            if value is None:
                value=await asyncio.to_thread(corridor,grid,job['spec'],current,target,guard=guard) if needed or nearest else dict(points=[],index=0,mode='course_retained')
                if value['points']:value['mode']='detour'
        if canal and value['points']:value.update(mode='canal',assumed_depth_m=10,measured_bathymetry=False)
        if coast_error:value['coastal_preference_error']=coast_error
        if guidance:value['mapped_guidance']=guidance
        if not value['points']:value['coastal_preference_error']=coast_error or 'no_verified_progressive_coastal_corridor'
        value.update(route_key=key,job=job['id'],source=job['manifest']['source'],policy=REGIONAL_POLICY,
                     mapped_land_checked=guard is not None)
        value['probe_position']=current
        return value
    async def close(self):
        if self.task:
            self.task.cancel();await asyncio.gather(self.task,return_exceptions=True)
        if self.index_task:
            self.index_task.cancel();await asyncio.gather(self.index_task,return_exceptions=True)
