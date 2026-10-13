"""Background regional lookahead for coarse maritime graph coast crossings.

Only plans geographic waypoints. Movement and local swept-hull approval remain
in native Gazebo. Downloading/planning never pauses the promotional world.
"""
import asyncio,math,time
from pyproj import Geod
GEO=Geod(ellps='WGS84')
REGIONAL_POLICY='buffered-water-v21-shallow-200m-coastal-320m'
COMPATIBLE_POLICIES={REGIONAL_POLICY}
PLANNING_BUDGET_SECONDS=660.
REGIONAL_BUFFER_M=200.
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
        # A safe frontier must advance towards the requested goal. A real
        # around-land detour supplies its own checked intermediate goal; an
        # arbitrary backward offset is not evidence of such a connection.
        progress=remaining-distance([lon,lat],target)
        if progress<min(50.,remaining*.1):continue
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
        self.progress=None;self.coastal_target=None
        self.recovery_extent=None;self.task_started=None
    def start_task(self,work):
        # The controller watchdog must not cancel terrain preparation before
        # its own bounded provider deadline. One budget includes all retries.
        self.task_started=time.monotonic()
        work_task=asyncio.create_task(work)
        work_task.add_done_callback(lambda task: task.exception() if not task.cancelled() else None)
        self.task=asyncio.create_task(asyncio.wait_for(work_task,PLANNING_BUDGET_SECONDS))
        # Cancellation before the wrapper starts must still close its work.
        self.task.add_done_callback(lambda _: work_task.cancel() if not work_task.done() else None)
    def recover_wait(self,state,preserve_pending=False):
        """Retry expired work; keep bounded pending or completed work consumable."""
        if preserve_pending and self.task is not None:
            elapsed=time.monotonic()-self.task_started if self.task_started is not None else PLANNING_BUDGET_SECONDS
            if self.task.done() or elapsed<PLANNING_BUDGET_SECONDS:
                state.setdefault('regional_planning',{}).update(elapsed_seconds=max(0.,elapsed),
                    budget_seconds=PLANNING_BUDGET_SECONDS)
                return dict(pending=True)
        if self.task:self.task.cancel()
        self.task=None;self.retry=0.;self.probed=None;self.progress=None
        previous=state.get('navigation_failsafe') or {}
        attempt=int(previous.get('attempts',0))+1
        reason=(state.get('regional_planning') or {}).get('reason') or 'checked_route_wait'
        self.discard(state,'checked_route_wait_stalled')
        self.recovery_extent=(6000.,20000.,40000.)[(attempt-1)%3]
        state['navigation_failsafe']=dict(status='replanning',attempts=attempt,
            checked_at=time.time(),extent_m=self.recovery_extent,
            reason=reason)
        (state.get('native',{}).get('navigation') or {}).pop('route_key',None)
        return dict(attempt=attempt,extent_m=self.recovery_extent)
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
        if plan.get('policy') not in COMPATIBLE_POLICIES:
            self.discard(state,'regional_clearance_policy_changed')
            return None
        from .course import passed_vertex
        # Persisted renewals from the previous coordinator may label their
        # entire retained runway as an exit. Only the appended tail is an exit.
        if plan.get('mode')=='shallow_exit' and plan.get('retained_checked_points',0):
            plan.setdefault('shallow_exit_start_index',plan['retained_checked_points'])
        exit_index=plan.get('shallow_exit_start_index')
        while plan['index']<len(plan['points']):
            index=plan['index'];point=plan['points'][index]
            if exit_index is not None:
                plan['mode']='shallow_exit' if index>=exit_index else plan.get('retained_mode','coastal')
            exit_handoff=exit_index is not None and index==exit_index-1
            reached=distance(state['position'],point)<(5 if exit_handoff or plan.get('mode') in ('port_departure','shallow_exit') else 12 if plan.get('mode')=='port_approach' else 80)
            previous=plan['points'][index-1] if index else plan.get('probe_position')
            passed=(not exit_handoff and plan.get('mode') in ('coastal','coastal_transfer','coastal_approach') and previous is not None
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
    def update(self,state,target,needed=False,terminal=False,retain_coast=False,progress_target=None,fixed=None,coastal_target=None):
        if fixed is not None:
            from .port_waypoints import ready,regional_plan
            if ready(fixed,state['waypoint'],state['position']):
                if self.task:self.task.cancel()
                self.task=None;self.retry=0;self.progress=None
                self.key=[state['lap'],state['leg'],state['waypoint']]
                state.pop('regional_waiting',None)
                state['regional_navigation']=regional_plan(fixed,state,REGIONAL_POLICY)
                state['regional_planning']=dict(status='ready',mode='fixed_port_approach',job=state['native']['job'])
                return fixed['coordinates'][state['waypoint']]
        if coastal_target!=self.coastal_target:
            # A newly discovered shoreline can change an optional graph goal
            # without changing the voyage's mandatory waypoint index.
            if self.task:self.task.cancel()
            self.task=None;self.retry=0.;self.probed=None
            self.coastal_target=list(coastal_target) if coastal_target is not None else None
        from posim_terrain.canal_override import in_region
        canal=in_region(state['position'])
        key=[state['lap'],state['leg'],state['waypoint']]
        departed=(state.get('last_stopover') or {}).get('port')
        if (not canal and departed and (state.get('last_stopover') or {}).get('departed_at')
                and distance(state['position'],[departed['longitude'],departed['latitude']])<5000.):
            return self.departure_update(state,target,key)

        from .port_approach import APPROACH_DISTANCE_M
        terminal=terminal and distance(state['position'],target)<APPROACH_DISTANCE_M
        state.pop('regional_waiting',None)
        if self.key!=key:
            if self.task:self.task.cancel()
            self.task=None;self.key=key;self.retry=0
            self.probed=None
            plan=state.get('regional_navigation') or {};old=plan.get('route_key')
            port_handoff=(terminal and plan.get('mode') in ('port_approach','port_approach_lookahead')
                          and plan.get('arrival_centre')==list(target))
            coast_handoff=(retain_coast and not terminal and not canal
                           and plan.get('mode') in ('coastal','coastal_transfer','coastal_approach'))
            if ((port_handoff or coast_handoff) and old and len(old)==3
                    and old[:2]==key[:2] and old[2]<key[2]
                    and plan.get('policy') in COMPATIBLE_POLICIES):
                plan['route_key']=list(key);plan.pop('continuation',None)
                state['regional_handoff']=dict(previous_key=old,route_key=list(key),
                    position=list(state['position']),reason='same_port_course_advance' if port_handoff else 'measured_coastal_shape_advance')
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
                from .coastal import COAST_DISCOVERY_M,MAXIMUM_COASTAL_DISTANCE_M
                separation,shore=self.coast_index.nearest(state['position'])
                if separation<COAST_DISCOVERY_M:
                    nearest=(separation,shore)
                    if separation>MAXIMUM_COASTAL_DISTANCE_M and hasattr(self.coast_index,'progressive_shore'):
                        nearest=self.coast_index.progressive_shore(state['position'],coastal_target or target)
                # Discovery proximity, not the presence of a forward candidate,
                # determines whether a coarse offshore vertex can be rounded.
                state['coastal_navigation']=dict(status='coastal_region' if separation<COAST_DISCOVERY_M else 'open_ocean',
                                                discovery_distance_m=separation)
        if canal and (state.get('regional_navigation') or {}).get('mode') not in (None,'canal'):
            self.discard(state,'suez_fine_grid')
        point=self.active_point(state,key)
        plan=state.get('regional_navigation') or {}
        # A mapped around-land guide must renew from its checked endpoint. It
        # cannot fall back to the original straight line across that land.
        guide_exhausted=(plan.get('route_key')==key and plan.get('mapped_guidance')
                         and plan.get('points') and plan['index']>=len(plan['points']))
        needed=needed or bool(guide_exhausted) or (plan.get('mode')=='shallow_exit' and plan.get('points') and plan['index']>=len(plan['points']))
        approach_runway=(plan.get('mode')=='port_approach_lookahead' and point is not None
                         and distance(state['position'],plan['probe_position'])<500.)
        prefer_port=terminal and (point is None or (plan.get('mode')!='port_approach' and not approach_runway))
        if terminal and point is not None and not prefer_port and not (self.task and self.task.done()):return point
        # Prepare the next coastal lookahead while the current checked plan
        # still has runway. Otherwise its end briefly falls back to an offshore
        # graph vertex and the ship turns out, then back when download finishes.
        renew_coast=(point is not None and nearest is not None and plan.get('mode') in ('coastal','coastal_transfer','coastal_approach')
                     and distance(state['position'],plan['points'][-1])<2500
                     and distance(state['position'],plan.get('probe_position',state['position']))>=1000
                     and (self.probed is None or distance(self.probed,state['position'])>=1000))
        from .coastal import WAYPOINT_FORMAT
        renew_coast=renew_coast or (point is not None and nearest is not None
                                    and plan.get('mode')=='coastal'
                                    and (plan.get('policy')!=REGIONAL_POLICY or not plan.get('onward_verified') or plan.get('waypoint_format')!=WAYPOINT_FORMAT))
        # Finish a known outward tail before asking for another continuation.
        renew_coast=renew_coast and plan.get('shallow_exit_start_index') is None
        coast_needed=nearest is not None and (self.probed is None or distance(self.probed,state['position'])>=3000)
        prefer_coast=(point is not None and coast_needed and not needed and
                      (state.get('regional_navigation') or {}).get('mode') not in ('coastal','coastal_transfer','coastal_approach'))
        if point is not None and not prefer_port and not prefer_coast and not renew_coast and not (self.task and self.task.done()):return point
        # A clear local segment is not proof that its regional destination is
        # reachable. Rebuild an expired/stalled goal even when the most recent
        # local segment did not itself need a detour or coast discovery.
        expired=state.get('regional_planning') or {}
        needed=needed or (expired.get('status')=='expired' and expired.get('reason') in
                         ('regional_clearance_policy_changed','regional_goal_progress_stalled','suez_fine_grid','predicted_stopping_corridor','checked_route_wait_stalled','shallow_water_clearance'))
        if self.task and self.task.done():
            try:
                value=self.task.result()
                connection=value.pop('continuation_of',None)
                if connection:
                    # Renew from the checked runway's END, retaining the
                    # remaining forward bends. Replanning from the moving hull
                    # can choose an already passed bay as a new coast entry.
                    if (plan.get('job')!=connection['job'] or plan.get('route_key')!=key
                            or connection['endpoint'] not in plan.get('points',[])):
                        raise RuntimeError('coastal_runway_changed')
                    end=plan['points'].index(connection['endpoint'])
                    if plan['index']>end:raise RuntimeError('coastal_runway_already_passed')
                    retained=plan['points'][plan['index']:end+1]
                    value['points']=retained+value['points'];value['index']=0
                    value['retained_checked_points']=len(retained)
                    if value.get('mode')=='shallow_exit':
                        value['shallow_exit_start_index']=len(retained)
                        value['retained_mode']=plan.get('mode','coastal')
                        value['mode']=value['retained_mode']
                    value['renewed_from']=connection
                    value['probe_position']=list(state['position'])
                    for metric in ('minimum_regional_clearance_m',):
                        if plan.get(metric) is not None and value.get(metric) is not None:
                            value[metric]=min(plan[metric],value[metric])
                # A failed optional preference must preserve a previously
                # checked detour while waiting for a better coastal region.
                coast_plan=(plan.get('route_key')==key and plan.get('mode') in ('coastal','coastal_transfer','coastal_approach')
                            and plan.get('policy') in COMPATIBLE_POLICIES)
                preserve_empty=(not value['points'] and (point is not None or
                                (coast_plan and nearest is not None and not terminal and not needed)))
                # An optional scenic renewal cannot replace a checked coastal
                # runway with an offshore water fallback and then turn back.
                preserve_fallback=(point is not None and coast_plan and nearest is not None and not terminal and not needed
                                   and value.get('mode')=='detour')
                preserve_existing=preserve_empty or preserve_fallback
                if not preserve_existing:state['regional_navigation']=value
                self.probed=list(value['probe_position']) if value.get('mode') not in ('coastal','coastal_transfer','coastal_approach') else None
                state['regional_planning']=dict(status='ready' if value['points'] else 'course_retained',
                                                job=value['job'],mode=value.get('mode','detour'),coastal_preference_error=value.get('coastal_preference_error'))
                if preserve_existing and coast_plan:
                    self.probed=None;self.retry=time.monotonic()+60
                    state['regional_planning']['status']='retrying'
            except (Exception,asyncio.CancelledError) as e:
                state['regional_planning']=dict(status='retrying',reason=str(e)[:160]);self.retry=time.monotonic()+60
            self.task=None
            point=self.active_point(state,key)
            if point is not None:
                if state['regional_navigation'].get('mode') in ('coastal','coastal_transfer','coastal_approach','port_approach'):
                    # Apply a completed background preference once, instead
                    # of first finishing an old corridor away from the coast.
                    (state.get('native',{}).get('navigation') or {}).pop('route_key',None)
                state['regional_planning'].update(status='ready',active_job=state['regional_navigation']['job']);return point
        coast_needed=nearest is not None and (self.probed is None or distance(self.probed,state['position'])>=3000)
        if (prefer_port or needed or coast_needed or renew_coast) and not self.task and time.monotonic()>=self.retry:
            state['regional_planning']=dict(status='background',mode='port_approach' if terminal else 'coastal' if coast_needed else 'detour')
            if terminal:work=self.prepare_port(state,target,key)
            elif renew_coast and point is not None:
                work=self.prepare_continuation(state,target,key,plan,extent=self.recovery_extent,
                    progress_target=progress_target,coastal_target=coastal_target)
            else:work=self.prepare(state,target,key,nearest,needed,extent=self.recovery_extent,
                    progress_target=progress_target,coastal_target=coastal_target)
            self.start_task(work)
            self.recovery_extent=None
        if point is not None:return point
        plan=state.get('regional_navigation') or {}
        if (plan.get('route_key')==key and plan.get('policy') in COMPATIBLE_POLICIES
                and (plan.get('mode') in ('coastal','coastal_transfer','coastal_approach') or plan.get('mapped_guidance')) and plan.get('points')
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
    def departure_update(self,state,target,key):
        """Commit bounded resident-water exit corridors toward the course."""
        state.pop('regional_waiting',None)
        if self.key!=key:
            if self.task:self.task.cancel()
            self.task=None;self.key=key;self.retry=0.
            state.pop('regional_navigation',None)
        point=self.active_point(state,key)
        if point is not None:return point
        if self.task and self.task.done():
            try:
                state['regional_navigation']=self.task.result()
                state['regional_planning']=dict(status='ready',mode='port_departure')
                (state.get('native',{}).get('navigation') or {}).pop('route_key',None)
                point=self.active_point(state,key)
            except Exception as error:
                self.retry=time.monotonic()+5.
                state['regional_planning']=dict(status='retrying',reason=str(error)[:160])
            self.task=None
            if point is not None:return point
        if self.task is None and time.monotonic()>=self.retry:
            self.start_task(self.prepare_departure(state,target,key))
            state['regional_planning']=dict(status='background',mode='port_departure')
        state['regional_waiting']=dict(reason='resident_departure_corridor_pending')
        return state['position']

    async def prepare_continuation(self,state,target,key,plan,extent=None,progress_target=None,coastal_target=None):
        endpoint=list(plan['points'][-1]);connection=dict(job=plan['job'],endpoint=endpoint)
        sample=dict(state,position=endpoint)
        nearest=None
        if self.coast_index is not None:
            from .coastal import COAST_DISCOVERY_M,MAXIMUM_COASTAL_DISTANCE_M
            separation,shore=self.coast_index.nearest(endpoint)
            if separation<COAST_DISCOVERY_M:
                nearest=(separation,shore)
                if separation>MAXIMUM_COASTAL_DISTANCE_M and hasattr(self.coast_index,'progressive_shore'):
                    nearest=self.coast_index.progressive_shore(endpoint,coastal_target or target)
        value=await self.prepare(sample,target,key,nearest,True,extent=extent,
            progress_target=progress_target,coastal_target=coastal_target)
        if not value.get('points'):raise RuntimeError('coastal_continuation_unavailable')
        value['continuation_of']=connection
        return value

    async def prepare_departure(self,state,target,key):
        from .port_approach import corridor
        from .land import guard_from_spec
        native=await self.rpc('/promotional/state');stream=await self.rpc('/stream/state')
        if stream.get('nonce')!=native['session_nonce']:raise RuntimeError('departure_world_changed')
        spec=(native.get('origin') or {}).get('spec')
        if spec is None:spec=(await self.rpc('/terrain/jobs/'+state['native']['job']))['spec']
        current=[native['geographic']['longitude'],native['geographic']['latitude']]
        heading,_,remaining=GEO.inv(*current,*target)
        goal=list(GEO.fwd(*current,heading,min(1000.,remaining))[:2])
        tiles=await asyncio.gather(*(self.rpc('/stream/tiles/'+tile['key']) for tile in stream['tiles']))
        guard=await asyncio.to_thread(guard_from_spec,spec,64000.)
        source='resident collision terrain';job_id=state['native']['job']
        try:value=await asyncio.to_thread(corridor,tiles,spec,current,goal,guard=guard,arrival=False)
        except RuntimeError:
            # A real around-land exit can exceed stationary resident coverage.
            # Resolve the exact course waypoint on a complete fine numeric field.
            heading,_,remaining=GEO.inv(*current,*target)
            if remaining>5000.:raise
            centre=list(GEO.fwd(*current,heading,remaining/2)[:2])
            job=await self.terrain_region(centre,6000.)
            grid=await self.rpc('/terrain/jobs/'+job['id']+'/files/terrain.json',timeout=20)
            region_guard=await asyncio.to_thread(guard_from_spec,job['spec'],64000.)
            value=await asyncio.to_thread(corridor,[grid],job['spec'],current,target,guard=region_guard,arrival=False)
            source=job['manifest']['source'];job_id=job['id']
        latest=await self.rpc('/promotional/state')
        if (latest['session_nonce']!=native['session_nonce']
                or (latest.get('origin') or {}).get('epoch',0)!=(native.get('origin') or {}).get('epoch',0)):
            raise RuntimeError('departure_frame_changed')
        value.update(route_key=key,mode='port_departure',job=job_id,
            policy=REGIONAL_POLICY,probe_position=current,source=source,mapped_land_checked=True)
        return value

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
            # Resolve the complete arrival on real regional triangles using
            # harbour hull/tracking clearance and the same endpoint screening.
            # The offshore 80 m search cells / 40 m buffer can close an entrance
            # that the resident harbour planner can safely traverse.
            heading,_,remaining=GEO.inv(*current,*target)
            centre=list(GEO.fwd(*current,heading,remaining/2)[:2])
            job=await self.terrain_region(centre,6000.)
            grid=await self.rpc('/terrain/jobs/'+job['id']+'/files/terrain.json',timeout=20)
            region_guard=None if in_region(current) else await asyncio.to_thread(guard_from_spec,job['spec'],64000.)
            value=await asyncio.to_thread(corridor,[grid],job['spec'],current,target,guard=region_guard)
            latest=await self.rpc('/promotional/state')
            if latest['session_nonce']!=native['session_nonce'] or (latest.get('origin') or {}).get('epoch',0)!=epoch:
                raise RuntimeError('port_frame_changed')
            value.update(route_key=key,job=job['id'],source=job['manifest']['source'],
                         policy=REGIONAL_POLICY,probe_position=current,mapped_land_checked=region_guard is not None,
                         scope='numeric regional harbour triangles; every native resident corridor rechecked')
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
            requested_at=time.time()
            job=await self.rpc('/terrain/jobs',dict(latitude=centre[1],longitude=centre[0],
                width_m=width,height_m=width,samples=257,source_policy='auto'),20)
            end=time.monotonic()+620
            while job['status'] in ('queued','running') and time.monotonic()<end:
                await asyncio.sleep(2);job=await self.rpc('/terrain/jobs/'+job['id'])
            if job['status']=='ready':return job
            # A long-lived API can have an older source fingerprint in memory
            # than its fresh terrain subprocess. This path reads numeric data
            # only, never starts that world's SDF. All commanded segments still
            # require current resident collision and mapped-land approval.
            if (job['status']=='outdated' and job.get('outdated_reasons')==['terrain_sources']
                    and job.get('created',0)>=requested_at-1.):
                return dict(job,numeric_grid_only=True)
            if width==6000 or job['status']!='failed':break
            try:
                sources=await self.rpc('/terrain/jobs/'+job['id']+'/files/source-providers.json')
            except Exception:break
            if (sources.get('baseline_status') or {}).get('status')!='unavailable':break
        raise RuntimeError('regional_terrain_unavailable')
    async def prepare(self,state,target,key,nearest=None,needed=True,extent=None,progress_target=None,coastal_target=None,fine_checked=False):
        from posim_terrain.canal_override import in_region
        canal=in_region(state['position'])
        if self.coastal and not canal and nearest is None and self.index_task is not None:
            # Policy migration can request a replan while the discovery index
            # is still warming. Do not commit another offshore detour first.
            try:
                index=await asyncio.shield(self.index_task)
                from .coastal import COAST_DISCOVERY_M,MAXIMUM_COASTAL_DISTANCE_M
                separation,shore=index.nearest(state['position'])
                if separation<COAST_DISCOVERY_M:
                    nearest=index.progressive_shore(state['position'],coastal_target or target) if separation>MAXIMUM_COASTAL_DISTANCE_M else (separation,shore)
            except Exception:pass  # Numeric collision routing remains available.
        course_target=target
        pos=list(state['position'])
        centre=pos
        if nearest and (extent is None or extent>6000) and 6000<nearest[0]<15000:
            # Keep the vessel inside a 20 km region while including a more
            # distant coast. The 1:10M discovery line never supplies depth.
            heading,_,_=GEO.inv(*pos,*nearest[1])
            centre=list(GEO.fwd(*pos,heading,min(6000.,nearest[0]-6000.))[:2])
        initial_extent=extent is None
        # Fine coastal triangles first: a 20 km grid can close navigable water
        # beside a headland and waste an entire provider/planner cycle.
        extent=extent or (4000 if canal else 6000 if nearest and nearest[0]<1500. else 20000)
        fine_checked=fine_checked or extent==6000
        if self.before_prepare is not None:await self.before_prepare()
        job=await self.terrain_region(centre,extent)
        grid=await self.rpc('/terrain/jobs/'+job['id']+'/files/terrain.json',timeout=20)
        guard=None
        if not canal:
            from .land import guard_from_spec
            guard=await asyncio.to_thread(guard_from_spec,job['spec'],64000.)
        current=list(state['position'])
        progress_target=progress_target or target;guidance=None
        coastal_goal=coastal_target if nearest is not None and coastal_target is not None else target
        value=None;coast_error=None
        if not canal and (nearest or needed):
            from .shallow import exit_corridor
            value=await asyncio.to_thread(exit_corridor,grid,job['spec'],current,progress_target,guard=guard)
        if value is None and not canal and (nearest or needed):
            from .coastal import coastal_corridor
            try:value=await asyncio.to_thread(coastal_corridor,grid,job['spec'],current,coastal_goal,guard=guard,progress_target=progress_target)
            except RuntimeError as error:coast_error=str(error)[:160]
        water_fallback=value if value is not None and value.get('mode')=='detour' else None
        if nearest and not fine_checked and (value is None or water_fallback is not None) and extent>6000 and nearest[0]<1500.:
            # A coarse buffered pixel can disconnect otherwise navigable coastal
            # water. Resolve the current shoreline before accepting a water exit.
            return await self.prepare(state,course_target,key,nearest,needed,extent=6000,
                                      progress_target=progress_target,coastal_target=coastal_target)
        if value is None or (nearest and water_fallback is not None):
            # A checked ordinary water path is a fallback, not proof that
            # approaching the nearby forward coast is unavailable.
            value=None
            current=list(state['position'])
            approach=None
            if nearest:
                from .coastal import coastal_approach
                approach=coastal_approach(current,coastal_goal,nearest[1],progress_target=progress_target)
            if approach:
                try:
                    # This small step is checked against real numeric depth,
                    # then checked again against the live Gazebo collision mesh.
                    value=await asyncio.to_thread(corridor,grid,job['spec'],current,approach,guard=guard)
                    endpoint=value['points'][-1]
                    from .coastal import MAXIMUM_COASTAL_DETOUR_RATIO
                    path_length=sum(distance(a,b) for a,b in zip([current,*value['points'][:-1]],value['points']))
                    if (distance(current,progress_target)-distance(endpoint,progress_target)<path_length/MAXIMUM_COASTAL_DETOUR_RATIO or
                        distance(current,coastal_goal)-distance(endpoint,coastal_goal)<500 or
                        distance(current,nearest[1])-distance(endpoint,nearest[1])<500):
                        value=None
                    else:
                        from .coastal import STANDOFF_M
                        value.update(mode='coastal_approach',preferred_standoff_m=STANDOFF_M)
                except RuntimeError as error:
                    coast_error=str(error)[:160]
            if value is None:
                try:
                    if water_fallback is not None:value=water_fallback
                    else:
                        if guard is not None and (needed or nearest):
                            from posim_terrain.terrain import projection
                            from .land_guidance import detour,local_target,horizon_target
                            _,forward,inverse=projection(job['spec'])
                            origin=forward.transform(*current);destination=forward.transform(*target)
                            # Optional global guidance must not veto a complete
                            # checked coastal path just because its straight ray
                            # falls on land. Ordinary detours still prove a full
                            # mapped-water connection before using a local prefix.
                            destination=horizon_target(guard,origin,destination)
                            points=await asyncio.to_thread(detour,guard,origin,destination)
                            if len(points)>1:
                                target=list(inverse.transform(*local_target(origin,points)))
                                guidance=dict(points=[list(inverse.transform(*p)) for p in points],
                                    course_target=list(course_target),
                                    scope='mapped land exclusion only; numeric depth checked locally')
                        value=await asyncio.to_thread(corridor,grid,job['spec'],current,target,guard=guard) if needed or nearest else dict(points=[],index=0,mode='course_retained')
                except RuntimeError as error:
                    # Coarse regional pixels can close a harbour entrance.
                    # Resolve it on a smaller real terrain grid before retrying;
                    # never substitute an arbitrary offshore frontier.
                    if not canal and not fine_checked and extent>6000 and str(error).startswith('regional_corridor_unavailable'):
                        return await self.prepare(state,course_target,key,nearest,needed,extent=6000,progress_target=progress_target,coastal_target=coastal_target)
                    if not canal and initial_extent and extent==6000:
                        return await self.prepare(state,course_target,key,nearest,needed,extent=20000,
                            progress_target=progress_target,coastal_target=coastal_target,fine_checked=True)
                    raise
                if value['points']:value['mode']='detour'
        if canal and value['points']:value.update(mode='canal',assumed_depth_m=10,measured_bathymetry=False)
        if coast_error:value['coastal_preference_error']=coast_error
        if guidance:value['mapped_guidance']=guidance
        if not value['points']:value['coastal_preference_error']=coast_error or 'no_verified_progressive_coastal_corridor'
        value.update(route_key=key,job=job['id'],source=job['manifest']['source'],policy=REGIONAL_POLICY,
                     mapped_land_checked=guard is not None)
        if not canal and value['points']:
            value.setdefault('shallow_water_buffer_m',REGIONAL_BUFFER_M)
            value.setdefault('shallow_boundary_depth_m',REGIONAL_DEPTH_M)
        value['probe_position']=current
        value['destination_progress_target']=list(progress_target)
        if coastal_target is not None and value.get('mode') in ('coastal','coastal_transfer','coastal_approach'):
            value['coastal_course_target']=list(coastal_target)
        return value
    async def close(self):
        if self.task:
            self.task.cancel();await asyncio.gather(self.task,return_exceptions=True)
        if self.index_task:
            self.index_task.cancel();await asyncio.gather(self.index_task,return_exceptions=True)
