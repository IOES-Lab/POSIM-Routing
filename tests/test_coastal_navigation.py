"""Coastal route contracts. Synthetic fixtures are not voyage evidence."""
import asyncio
import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
from posim_routing.coastal import CoastIndex, coastal_corridor, coastal_approach, MINIMUM_DEPTH_M, REGIONAL_BUFFER_M, STANDOFF_M, MAXIMUM_COASTAL_DETOUR_RATIO, MAXIMUM_COASTAL_DISTANCE_M
from posim_routing.navigation import TerrainField, DetourPlanner
from posim_routing.planning import CachedTerrain
from posim_routing.regional import RegionalNavigator,REGIONAL_POLICY,distance,corridor
from posim_terrain.terrain import projection

SPEC = dict(latitude=35., longitude=129., width_m=20000, height_m=20000, samples=257)
_, FORWARD, INVERSE = projection(SPEC)


def geo(x, y):
    return list(INVERSE.transform(x, y))


def grid(headland=False):
    axis = np.linspace(-10000, 10000, 257)
    x, y = np.meshgrid(axis, axis)
    land = x <= 0
    if headland:
        land |= (x <= 1800) & (y > -1000) & (y < 1500)
    heights = np.where(land, 20., -15.)
    return dict(xs=axis.tolist(), ys=axis.tolist(), heights=heights.tolist())


class CoastalPlanning(unittest.TestCase):
    def assert_swept_safe(self, terrain, start, plan):
        field = CachedTerrain(TerrainField(terrain['xs'], terrain['ys'], terrain['heights']))
        origin = [*FORWARD.transform(*start), 0.]
        validator = DetourPlanner(field, origin, True, maximum_radius=18000)
        validator.radius = REGIONAL_BUFFER_M
        validator.clearance = MINIMUM_DEPTH_M
        points = [origin] + [[*FORWARD.transform(*p), 0.] for p in plan['points']]
        self.assertTrue(all(validator.segment(a, b) for a, b in zip(points, points[1:])))

    def assert_coastal_band(self, terrain, start, plan):
        from scipy.spatial import cKDTree
        x, y = np.meshgrid(terrain['xs'], terrain['ys'])
        land = np.asarray(terrain['heights']) >= 0
        shore = cKDTree(np.column_stack((x[land], y[land])))
        points = [FORWARD.transform(*start)] + [FORWARD.transform(*p)
                  for p in plan['points'] + plan['onward_points']]
        if shore.query(points[0])[0] > MAXIMUM_COASTAL_DISTANCE_M:
            entry = np.array(FORWARD.transform(*plan['coastal_band_entry']))
            joined = next(i for i, point in enumerate(points) if np.linalg.norm(np.array(point)-entry) < 1.)
            points = points[joined:]
        samples = np.concatenate([np.linspace(a, b, max(2, int(np.linalg.norm(np.array(b)-a)/10)+1))
                                  for a, b in zip(points, points[1:])])
        self.assertLessEqual(float(shore.query(samples)[0].max()), MAXIMUM_COASTAL_DISTANCE_M)
        self.assertEqual(plan['maximum_coastal_distance_m'], 1500.)

    def test_small_island_transfers_to_a_progressive_mainland_band(self):
        terrain=grid();x,y=np.meshgrid(terrain['xs'],terrain['ys'])
        terrain['heights']=np.where((x<=0)|((abs(x-5000)<250)&(abs(y)<250)),20.,-15.).tolist()
        start=geo(5000,-650);target=geo(800,-9000)
        plan=coastal_corridor(terrain,SPEC,start,target)
        self.assertEqual(plan['mode'],'coastal_transfer')
        self.assertTrue(plan['coastal_transfer_verified'])
        entry_x,_=FORWARD.transform(*plan['coastal_band_entry'])
        self.assertLess(entry_x,1500.)
        self.assertEqual(plan['coastal_band_reference'],plan['coastal_band_entry'])
        self.assert_swept_safe(terrain,start,plan)
        self.assertLessEqual(plan['coastal_through_length_m'],2*plan['shortest_checked_length_m'])

    def test_simplified_coastal_segments_and_onward_runway_stay_in_band(self):
        for terrain in (grid(), grid(headland=True)):
            start = geo(800, -4000)
            plan = coastal_corridor(terrain, SPEC, start, geo(800, 9000))
            self.assertEqual(plan['mode'], 'coastal')
            self.assert_coastal_band(terrain, start, plan)

    def test_doubled_detour_budget_can_join_previously_rejected_coast(self):
        start, target = geo(5000, -4000), geo(6500, 9000)
        plan = coastal_corridor(grid(), SPEC, start, target)
        self.assertEqual(plan['mode'], 'coastal')
        self.assertGreater(plan['coastal_through_length_m']/plan['shortest_checked_length_m'], 1.5)
        self.assertLessEqual(plan['coastal_through_length_m']/plan['shortest_checked_length_m'], 2.)
        self.assert_coastal_band(grid(), start, plan)

    def test_clear_coastal_run_does_not_keep_grid_staircases(self):
        for amplitude, period in ((0., 1000.), (500., 2000.)):
            with self.subTest(shore_amplitude=amplitude):
                terrain = grid()
                x, y = np.meshgrid(terrain['xs'], terrain['ys'])
                terrain['heights'] = np.where(
                    x <= amplitude*np.sin(y/period*2*np.pi), 20., -15.).tolist()
                start = geo(1200, -4000)
                plan = coastal_corridor(terrain, SPEC, start, geo(1200, 9000))
                self.assertEqual(plan['mode'], 'coastal')
                through = plan['points'] + plan['onward_points']
                # A clear run needs bounded segment ends, not one turn per
                # ten grid cells or every small shoreline indentation.
                self.assertLessEqual(len(through), 4)
                self.assertLess(plan['coastal_through_length_m'], 14500.)
                self.assertTrue(all(distance(a, b) <= 7501.
                                    for a, b in zip([start]+through[:-1], through)))
                self.assert_coastal_band(terrain, start, plan)
                self.assert_swept_safe(terrain, start, dict(points=through))

    def test_safe_backward_frontier_is_rejected(self):
        axis=np.linspace(-10000,10000,257)
        x,y=np.meshgrid(axis,axis)
        terrain=dict(xs=axis,ys=axis,heights=np.where(y>300.,-1.,-15.))
        # Only goals facing away from the shallow northern target are safe.
        with self.assertRaisesRegex(RuntimeError,'regional_corridor_unavailable'):
            corridor(terrain,SPEC,geo(0,0),geo(0,1000))

    def test_forward_corridor_keeps_goal_progress(self):
        start,target=geo(3000,-4000),geo(3000,5000)
        plan=corridor(grid(),SPEC,start,target)
        self.assertGreater(distance(start,target)-distance(plan['points'][-1],target),7000.)
        self.assert_swept_safe(grid(),start,plan)

    def test_stays_near_coast_and_progresses(self):
        terrain = grid()
        start, target = geo(800, -4000), geo(800, 9000)
        plan = coastal_corridor(terrain, SPEC, start, target)
        self.assertIsNotNone(plan)
        self.assertEqual(plan['mode'], 'coastal')
        self.assertLess(abs(plan['endpoint_standoff_m']-STANDOFF_M), 200)
        self.assertGreater(plan['progress_to_course_waypoint_m'], 1000)
        self.assertLessEqual(plan['coastal_through_length_m'], plan['shortest_checked_length_m']*MAXIMUM_COASTAL_DETOUR_RATIO)
        self.assert_swept_safe(terrain, start, plan)

    def test_start_outside_shallow_buffer_has_a_checked_coastal_connection(self):
        terrain = grid()
        start = geo(500, -4000)
        plan = coastal_corridor(terrain, SPEC, start, geo(800, 9000))
        self.assertIsNotNone(plan)
        self.assertLess(plan['endpoint_standoff_m'], STANDOFF_M+100.)
        self.assert_swept_safe(terrain, start, plan)

    def test_80m_closer_preference_keeps_checked_route_and_onward_progress(self):
        from unittest.mock import patch
        terrain = grid()
        start, target = geo(800, -4000), geo(800, 9000)
        with patch('posim_routing.coastal.STANDOFF_M', 400.):
            before = coastal_corridor(terrain, SPEC, start, target)
        after = coastal_corridor(terrain, SPEC, start, target)
        self.assertEqual(after['mode'], 'coastal')
        self.assertEqual(after['preferred_standoff_m']-before['preferred_standoff_m'], -80.)
        shift = after['endpoint_standoff_m']-before['endpoint_standoff_m']
        self.assertGreaterEqual(shift, -160.)
        self.assertLessEqual(shift, 0.)
        self.assertEqual(after['maximum_coastal_distance_m'], 1500.)
        self.assert_coastal_band(terrain, start, after)
        self.assert_swept_safe(terrain, start, after)

    def test_strict_budget_retains_shorter_water_route(self):
        terrain = grid()
        start, target = geo(5000, -4000), geo(6500, 9000)
        from unittest.mock import patch
        with patch('posim_routing.coastal.MAXIMUM_COASTAL_DETOUR_RATIO', 1.5):
            plan = coastal_corridor(terrain, SPEC, start, target)
        self.assertEqual(plan['mode'],'detour')
        self.assertEqual(plan['coastal_preference_reason'],'shorter_checked_water_route')
        self.assertGreater(distance(start,target)-distance(plan['points'][-1],target), 7000.)
        self.assert_swept_safe(terrain, start, plan)

    def test_headland_cannot_be_shortcut(self):
        terrain = grid(headland=True)
        start = geo(800, -4000)
        # The relaxed coastal preference still requires going around land.
        target = geo(800, 9000)
        plan = coastal_corridor(terrain, SPEC, start, target)
        self.assertEqual(plan['mode'],'coastal')
        self.assertLessEqual(plan['coastal_through_length_m'], plan['shortest_checked_length_m']*MAXIMUM_COASTAL_DETOUR_RATIO)
        through = plan['points']+plan['onward_points']
        points = np.array([FORWARD.transform(*p) for p in through])
        self.assertGreater(points[:, 0].max(), 1800+REGIONAL_BUFFER_M)
        self.assert_swept_safe(terrain, start, dict(points=through))

    def test_closed_bay_requires_an_onward_water_exit(self):
        terrain = grid()
        x,y = np.meshgrid(terrain['xs'],terrain['ys'])
        # Bay opens east; its western head is nearer the southwest goal.
        land = (x < -2000) | ((y < -1000) & (x < 5000)) | ((y > 4500) & (x < 5000))
        terrain['heights'] = np.where(land,20.,-15.).tolist()
        start,target = geo(1000,2000),geo(-20000,-20000)
        plan = coastal_corridor(terrain,SPEC,start,target)
        self.assertEqual(plan['mode'],'coastal')
        self.assert_coastal_band(terrain, start, plan)
        self.assertLessEqual(plan['coastal_through_length_m'], 2*plan['shortest_checked_length_m'])
        end_x,end_y = FORWARD.transform(*plan['points'][-1])
        self.assertGreater(end_x,5000)
        self.assertLess(end_y,-1000)
        self.assert_swept_safe(terrain,start,plan)

    def test_long_onward_runs_use_segment_ends_and_keep_the_exit(self):
        terrain = grid()
        x,y = np.meshgrid(terrain['xs'],terrain['ys'])
        terrain['heights'] = np.where((x<=0)&(y<=2000),20.,-15.).tolist()
        start = geo(800, -4000)
        plan = coastal_corridor(terrain, SPEC, start, geo(6500, 9000))
        self.assertIsNotNone(plan)
        self.assertGreater(distance(plan['points'][-1], plan['onward_exit']), 5000.)
        self.assertTrue(plan['onward_verified'])
        self.assertGreater(len(plan['onward_points']), 0)
        self.assertGreater(distance(plan['onward_points'][-1], plan['onward_exit']), 3000.)
        self.assert_coastal_band(terrain, start, plan)
        self.assert_swept_safe(terrain, start, dict(points=plan['points']+plan['onward_points']))

    def test_shallow_and_unknown_coast_is_not_a_destination(self):
        terrain = grid()
        heights = np.array(terrain['heights'])
        x = np.array(terrain['xs'])
        heights[:, (x > 0) & (x < 1700)] = -1.
        terrain['heights'] = heights.tolist()
        plan=coastal_corridor(terrain, SPEC, geo(3000,-4000),geo(6500,9000))
        # The bank blocks the coastal band around real land; use checked open water.
        self.assertIsNone(plan)
        water = corridor(terrain, SPEC, geo(3000,-4000),geo(6500,9000))
        self.assert_swept_safe(terrain,geo(3000,-4000),water)
        self.assertGreater(FORWARD.transform(*water['points'][-1])[0],1900.)
        heights[:, (x > 0) & (x < 1700)] = np.nan
        terrain['heights'] = heights.tolist()
        self.assertIsNone(coastal_corridor(terrain, SPEC, geo(3000, -4000), geo(6500, 9000)))

    def test_submerged_banks_do_not_attract_coastal_waypoints(self):
        for actual_land in (False, True):
            with self.subTest(actual_land=actual_land):
                terrain = grid()
                x, y = np.meshgrid(terrain['xs'], terrain['ys'])
                heights = np.where(x <= 0, -1., -15.)
                if actual_land:
                    heights[x <= -6000] = 20.
                terrain['heights'] = heights.tolist()
                # The safe side of this bank is open water, kilometres from
                # real land. It must not become a scenic coastline to follow.
                self.assertIsNone(coastal_corridor(
                    terrain, SPEC, geo(800, -4000), geo(800, 9000)))
                water = corridor(terrain, SPEC, geo(800, -4000), geo(800, 9000))
                self.assertEqual(len(water['points']), 1)
                end_x, _ = FORWARD.transform(*water['points'][0])
                self.assertLess(abs(end_x-800.), 1.)

    def test_two_point_five_metre_coast_is_available_to_wamv(self):
        terrain=grid();heights=np.array(terrain['heights'])
        heights[heights<0]=-2.5;terrain['heights']=heights.tolist()
        start=geo(800,-4000);plan=coastal_corridor(terrain,SPEC,start,geo(800,9000))
        self.assertIsNotNone(plan)
        self.assertGreaterEqual(plan['minimum_regional_clearance_m'],2.)
        self.assertLess(plan['endpoint_standoff_m'],1000)
        self.assert_swept_safe(terrain,start,plan)

    def test_ocean_retains_course_and_near_goal_requires_exact_arrival(self):
        terrain = grid()
        terrain['heights'] = np.full((257, 257), -50.).tolist()
        self.assertIsNone(coastal_corridor(terrain, SPEC, geo(1000, -4000), geo(6000, 6000)))
        self.assertIsNone(coastal_corridor(grid(), SPEC, geo(800, 0), geo(1800, 0)))

    def test_backward_only_coast_is_not_selected(self):
        # An eastbound departure must not be pulled back west merely for land.
        self.assertIsNone(coastal_corridor(grid(), SPEC, geo(4500, 0), geo(20000, 0)))

    def test_coarse_waypoint_cannot_pull_coastal_tail_away_from_port(self):
        terrain = grid()
        start, target = geo(800, -4000), geo(800, 9000)
        self.assertIsNotNone(coastal_corridor(terrain, SPEC, start, target))
        plan=coastal_corridor(terrain, SPEC, start, target,progress_target=geo(9000, -5000))
        self.assertEqual(plan['mode'],'detour')
        self.assert_swept_safe(terrain,start,plan)

    def test_real_discovery_index_wraps_longitude(self):
        index = CoastIndex()
        distance, _ = index.nearest([129.08468, 35.07446])
        self.assertLess(distance, 5000)
        a = index.nearest([179.9, -16.5])
        b = index.nearest([-180.1, -16.5])
        self.assertAlmostEqual(a[0], b[0], places=6)

    def test_distant_coast_approach_advances_without_changing_course_target(self):
        from posim_routing.regional import distance
        start, target, shore = geo(30000, 0), geo(0, -90000), geo(0, -60000)
        candidate=coastal_approach(start,target,shore)
        self.assertIsNotNone(candidate)
        self.assertLessEqual(distance(start,candidate),7500.01)
        self.assertGreater(distance(start,target)-distance(candidate,target),500)
        self.assertGreater(distance(start,shore)-distance(candidate,shore),500)
        self.assertEqual(target,geo(0,-90000))
        self.assertIsNone(coastal_approach(geo(800,0),target,geo(0,0)))
        self.assertIsNone(coastal_approach(geo(30000,0),geo(90000,0),geo(0,0)))
        relaxed=coastal_approach(start,target,geo(0,-12000))
        self.assertIsNotNone(relaxed)
        travel=distance(start,relaxed)
        progress=distance(start,target)-distance(relaxed,target)
        self.assertGreaterEqual(progress,travel/MAXIMUM_COASTAL_DETOUR_RATIO)
        self.assertLess(progress,travel/1.15)
        self.assertIsNone(coastal_approach(start,target,shore,progress_target=geo(90000,0)))

    def test_real_progressive_shore_excludes_backward_coast(self):
        from posim_routing.regional import distance
        index=CoastIndex();start=[122.72,29.85];target=[118.0835,24.4363]
        nearest=index.progressive_shore(start,target)
        self.assertIsNotNone(nearest)
        self.assertLess(nearest[0],80000)
        self.assertGreater(distance(start,target)-distance(nearest[1],target),999)


class BackgroundPlanning(unittest.IsolatedAsyncioTestCase):
    async def test_checked_coast_renews_even_when_global_ray_is_on_land(self):
        from unittest.mock import patch
        async def rpc(path,body=None,timeout=10):
            if path.endswith('/terrain.json'):return grid()
            return dict(id='fixture',status='ready',spec=SPEC,manifest=dict(source='fixture'))
        navigator=RegionalNavigator(rpc,coastal=True)
        sample=dict(position=geo(800,0))
        checked=dict(points=[geo(800,5000)],index=0,mode='coastal',onward_verified=True)
        with patch('posim_routing.coastal.coastal_corridor',return_value=checked), \
             patch('posim_routing.land_guidance.horizon_target',side_effect=RuntimeError('mapped_guidance_horizon_blocked')) as horizon:
            plan=await navigator.prepare(sample,geo(-8000,90000),[0,0,1],(800.,geo(0,0)),True)
        self.assertEqual(plan['mode'],'coastal');horizon.assert_not_called()
        await navigator.close()

    async def test_wait_failsafe_cancels_stale_work_and_changes_resolution_without_moving(self):
        from unittest.mock import patch
        async def rpc(*args,**kwargs):await asyncio.Event().wait()
        class Near:
            def nearest(self,position):return 800.,geo(0,0)
        key=[0,0,1];position=geo(800,3000)
        sample=dict(lap=0,leg=0,waypoint=1,position=position,native=dict(navigation=dict(route_key=key)),
            regional_navigation=dict(points=[position],index=1,route_key=key,mode='coastal',
                policy=REGIONAL_POLICY,job='checked',probe_position=geo(800,0)))
        navigator=RegionalNavigator(rpc,coastal=True);navigator.coast_index=Near();navigator.key=key
        navigator.task=asyncio.create_task(rpc());old=navigator.task
        self.assertEqual(navigator.recover_wait(sample)['extent_m'],6000.)
        await asyncio.gather(old,return_exceptions=True)
        self.assertTrue(old.cancelled());self.assertNotIn('route_key',sample['native']['navigation'])
        self.assertEqual(sample['position'],position);self.assertNotIn('regional_navigation',sample)
        navigator.update(sample,geo(6500,9000));self.assertIsNotNone(navigator.task)
        self.assertEqual(sample['navigation_failsafe']['attempts'],1)
        self.assertEqual(navigator.recover_wait(sample)['extent_m'],20000.)
        self.assertEqual(navigator.recover_wait(sample)['extent_m'],40000.)
        await navigator.close()

    async def test_wait_watch_preserves_pending_work_until_its_own_deadline(self):
        from unittest.mock import patch
        from posim_routing.regional import PLANNING_BUDGET_SECONDS
        gate=asyncio.Event()
        async def prepare():await gate.wait();return {'points': [[1,2]]}
        navigator=RegionalNavigator(None);sample=dict(regional_planning=dict(status='background'))
        navigator.start_task(prepare());task=navigator.task
        await asyncio.sleep(0)
        with patch('posim_routing.regional.time.monotonic',return_value=navigator.task_started+120):
            self.assertEqual(navigator.recover_wait(sample,preserve_pending=True),dict(pending=True))
        self.assertIs(navigator.task,task);self.assertFalse(task.cancelled())
        self.assertNotIn('navigation_failsafe',sample)
        gate.set();await task
        self.assertEqual(navigator.recover_wait(sample,preserve_pending=True),dict(pending=True))
        self.assertEqual(task.result()['points'],[[1,2]])
        await navigator.close()

    async def test_wait_watch_replaces_work_that_exceeds_total_budget(self):
        from unittest.mock import patch
        from posim_routing.regional import PLANNING_BUDGET_SECONDS
        async def prepare():await asyncio.Event().wait()
        navigator=RegionalNavigator(None);sample={}
        navigator.start_task(prepare());task=navigator.task;await asyncio.sleep(0)
        with patch('posim_routing.regional.time.monotonic',return_value=navigator.task_started+PLANNING_BUDGET_SECONDS+1):
            value=navigator.recover_wait(sample,preserve_pending=True)
        await asyncio.gather(task,return_exceptions=True)
        self.assertTrue(task.cancelled());self.assertEqual(value['attempt'],1)
        self.assertEqual(value['extent_m'],6000.)
        await navigator.close()

    async def test_near_shore_uses_fine_grid_first_and_grows_only_once(self):
        from unittest.mock import patch
        widths=[]
        async def rpc(path,body=None,timeout=10):
            if body:
                widths.append(body['width_m'])
                return dict(id='fixture',status='ready',spec=SPEC,manifest=dict(source='fixture'))
            return grid()
        navigator=RegionalNavigator(rpc,coastal=True)
        start=geo(800,0);target=geo(800,90000)
        checked=dict(points=[geo(800,5000)],index=0,mode='coastal')
        with patch('posim_routing.coastal.coastal_corridor',return_value=checked):
            await navigator.prepare(dict(position=start),target,[0,0,1],(800.,geo(0,0)))
        self.assertEqual(widths,[6000])
        widths.clear()
        with patch('posim_routing.coastal.coastal_corridor',return_value=None), \
             patch('posim_routing.coastal.coastal_approach',return_value=None), \
             patch('posim_routing.land_guidance.detour',return_value=[]), \
             patch('posim_routing.regional.corridor',side_effect=RuntimeError('regional_corridor_unavailable')):
            with self.assertRaisesRegex(RuntimeError,'regional_corridor_unavailable'):
                await navigator.prepare(dict(position=start),target,[0,0,1],(800.,geo(0,0)))
        self.assertEqual(widths,[6000,20000])
        await navigator.close()

    async def test_larger_shallow_buffer_requires_rechecking_old_runway(self):
        from unittest.mock import patch
        async def rpc(*args,**kwargs):await asyncio.Event().wait()
        class Near:
            def nearest(self,position):return 400.,geo(0,0)
        key=[0,0,1];start=geo(400,0);next_point=geo(400,2000)
        sample=dict(lap=0,leg=0,waypoint=1,position=start,native={},
            regional_navigation=dict(points=[next_point],index=0,mode='coastal',route_key=key,
                policy='buffered-water-v18-coastal-350m',job='checked',probe_position=start))
        navigator=RegionalNavigator(rpc,coastal=True);navigator.coast_index=Near();navigator.key=key
        self.assertEqual(navigator.update(sample,geo(400,90000)),start)
        self.assertIn('regional_waiting',sample);self.assertIsNotNone(navigator.task)
        await asyncio.sleep(0);await navigator.close()

    async def test_coastal_renewal_is_prepared_from_runway_end(self):
        from unittest.mock import AsyncMock
        navigator=RegionalNavigator(None,coastal=True)
        start,end,new=geo(800,0),geo(800,3000),geo(800,5500)
        sample=dict(position=start)
        navigator.prepare=AsyncMock(return_value=dict(points=[new],index=0,mode='coastal'))
        value=await navigator.prepare_continuation(sample,geo(800,90000),[0,0,1],
            dict(points=[end],job='runway'))
        self.assertEqual(navigator.prepare.call_args.args[0]['position'],end)
        self.assertEqual(sample['position'],start)
        self.assertEqual(value['continuation_of'],dict(job='runway',endpoint=end))

    async def test_completed_renewal_retains_forward_bends_before_continuation(self):
        from posim_routing.coastal import WAYPOINT_FORMAT
        key=[0,0,1];start,first,end,new=geo(800,1000),geo(800,2000),geo(800,3000),geo(800,5500)
        navigator=RegionalNavigator(None);navigator.key=key
        sample=dict(lap=0,leg=0,waypoint=1,position=start,native={},
            regional_navigation=dict(points=[geo(800,0),first,end],index=1,mode='coastal',
                route_key=key,policy=REGIONAL_POLICY,job='runway',probe_position=geo(800,0)))
        async def result():return dict(points=[new],index=0,mode='coastal',route_key=key,
            policy=REGIONAL_POLICY,job='renewal',probe_position=end,onward_verified=True,
            waypoint_format=WAYPOINT_FORMAT,continuation_of=dict(job='runway',endpoint=end))
        navigator.task=asyncio.create_task(result());await navigator.task
        self.assertEqual(navigator.update(sample,geo(800,90000)),first)
        self.assertEqual(sample['regional_navigation']['points'],[first,end,new])
        self.assertEqual(sample['regional_navigation']['retained_checked_points'],2)
        self.assertNotIn('regional_waiting',sample)
        await navigator.close()

    async def test_outward_tail_does_not_relabel_retained_forward_runway(self):
        key=[0,0,1];start,first,end,exit_point=geo(800,1000),geo(800,2000),geo(800,3000),geo(1000,3000)
        navigator=RegionalNavigator(None);navigator.key=key
        sample=dict(lap=0,leg=0,waypoint=1,position=start,native={},
            regional_navigation=dict(points=[first,end],index=0,mode='coastal',route_key=key,
                policy=REGIONAL_POLICY,job='runway',probe_position=geo(800,0)))
        async def result():return dict(points=[exit_point],index=0,mode='shallow_exit',route_key=key,
            policy=REGIONAL_POLICY,job='exit',probe_position=end,
            continuation_of=dict(job='runway',endpoint=end))
        navigator.task=asyncio.create_task(result());await navigator.task
        self.assertEqual(navigator.update(sample,geo(800,90000)),first)
        plan=sample['regional_navigation']
        self.assertEqual(plan['mode'],'coastal');self.assertEqual(plan['shallow_exit_start_index'],2)
        sample['position']=first
        self.assertEqual(navigator.active_point(sample,key),end)
        sample['position']=geo(800,2950)
        self.assertEqual(navigator.active_point(sample,key),end)
        self.assertEqual(plan['mode'],'coastal')
        sample['position']=end
        self.assertEqual(navigator.active_point(sample,key),exit_point)
        self.assertEqual(plan['mode'],'shallow_exit');self.assertEqual(plan['index'],2)
        await navigator.close()

    async def test_saved_exit_with_retained_points_migrates_only_its_phase(self):
        key=[0,0,1];first,end,exit_point=geo(800,2000),geo(800,3000),geo(1000,3000)
        plan=dict(points=[first,end,exit_point],index=0,mode='shallow_exit',route_key=key,
            policy=REGIONAL_POLICY,job='saved',probe_position=geo(800,0),retained_checked_points=2)
        sample=dict(position=geo(800,1000),native={},regional_navigation=plan)
        navigator=RegionalNavigator(None)
        self.assertEqual(navigator.active_point(sample,key),first)
        self.assertEqual(plan['mode'],'coastal');self.assertEqual(plan['points'],[first,end,exit_point])
        sample['position']=first
        self.assertEqual(navigator.active_point(sample,key),end)
        sample['position']=end
        self.assertEqual(navigator.active_point(sample,key),exit_point)
        self.assertEqual(plan['mode'],'shallow_exit')
        await navigator.close()

    async def test_exhausted_coast_accepts_checked_replacement_water_route(self):
        async def rpc(*args,**kwargs):await asyncio.Event().wait()
        class Near:
            def nearest(self,position):return 800.,geo(0,0)
        key=[0,0,1];end=geo(800,3000);new=geo(1000,5000)
        sample=dict(lap=0,leg=0,waypoint=1,position=end,native={},
            regional_navigation=dict(points=[end],index=1,mode='coastal',route_key=key,
                policy=REGIONAL_POLICY,job='old',probe_position=geo(800,0)))
        navigator=RegionalNavigator(rpc,coastal=True);navigator.coast_index=Near();navigator.key=key
        async def ready():return dict(points=[new],index=0,mode='detour',route_key=key,
            policy=REGIONAL_POLICY,job='checked-water',probe_position=end)
        navigator.task=asyncio.create_task(ready());await navigator.task
        self.assertEqual(navigator.update(sample,geo(6500,9000)),new)
        self.assertEqual(sample['regional_navigation']['job'],'checked-water')
        self.assertNotIn('regional_waiting',sample)
        await navigator.close()

    async def asyncSetUp(self):
        from unittest.mock import patch
        import shapely
        from posim_routing.land import LandGuard
        # These RPC fixtures carry synthetic numeric terrain, not map data.
        fixture=LandGuard(shapely.GeometryCollection(),shapely.box(-100000,-100000,100000,100000))
        mock=patch('posim_routing.land.guard_from_spec',return_value=fixture)
        mock.start();self.addCleanup(mock.stop)

    async def test_checked_coastal_onward_goal_preserves_port_and_shape_index(self):
        from unittest.mock import patch
        async def rpc(path,body=None,timeout=10):
            if path.endswith('/terrain.json'):return grid()
            return dict(id='fixture',status='ready',spec=SPEC,manifest=dict(source='fixture'))
        sample=dict(lap=0,leg=0,waypoint=1,position=geo(800,-4000))
        vertex,onward,port=geo(30000,0),geo(800,9000),geo(-8000,50000)
        navigator=RegionalNavigator(rpc,coastal=True)
        with patch('posim_routing.coastal.coastal_corridor',return_value=dict(points=[geo(800,5000)],index=0,mode='coastal')) as checked:
            plan=await navigator.prepare(sample,vertex,[0,0,1],(1000.,geo(0,0)),False,
                                         progress_target=port,coastal_target=onward)
        self.assertEqual(checked.call_args.args[3],onward)
        self.assertEqual(checked.call_args.kwargs['progress_target'],port)
        self.assertEqual(plan['coastal_course_target'],onward)
        self.assertEqual(sample['waypoint'],1)
        await navigator.close()

    async def test_nearby_coast_stays_a_coastal_region_without_a_forward_sample(self):
        # After a fixed departure, the coarse target points offshore. Coast
        # discovery must still enable rounding before seeking its next shore.
        class Near:
            def nearest(self, position):return 12000., geo(0, 0)
            def progressive_shore(self, position, target):return None
        navigator=RegionalNavigator(lambda *args: None, coastal=True)
        navigator.coast_index=Near();position=geo(8000, 0)
        navigator.probed=position;sample=dict(lap=0,leg=0,waypoint=1,position=position,native={})
        self.assertEqual(navigator.update(sample,geo(30000,0)),geo(30000,0))
        self.assertEqual(sample['coastal_navigation']['status'],'coastal_region')
        self.assertIsNone(navigator.task)
        await navigator.close()

    async def test_checked_water_fallback_does_not_suppress_shore_approach(self):
        from unittest.mock import patch
        async def rpc(path,body=None,timeout=10):
            if path.endswith('/terrain.json'):return grid()
            return dict(id='fixture',status='ready',spec=SPEC,manifest=dict(source='fixture'))
        position=geo(5000,-4000);target=geo(6500,9000);near=(10000.,geo(0,5000))
        water=dict(points=[geo(6500,5000)],index=0,mode='detour')
        endpoint=geo(1000,3000);checked=dict(points=[endpoint],index=0)
        navigator=RegionalNavigator(rpc,coastal=True)
        with patch('posim_routing.coastal.coastal_corridor',return_value=water), \
             patch('posim_routing.coastal.coastal_approach',return_value=endpoint), \
             patch('posim_routing.regional.corridor',return_value=checked) as approach:
            value=await navigator.prepare(dict(position=position),target,[0,0,1],near,False)
        approach.assert_called_once()
        self.assertEqual(value['mode'],'coastal_approach')
        self.assertEqual(value['points'],[endpoint])
        self.assertGreater(distance(position,near[1])-distance(endpoint,near[1]),500.)
        await navigator.close()

    async def test_nonempty_optional_water_fallback_preserves_coastal_runway(self):
        class Near:
            def nearest(self,position):return 800.,geo(0,0)
        async def rpc(*args,**kwargs):raise AssertionError('retry must be throttled')
        key=[0,0,1];position=geo(800,3000);end=geo(800,4000)
        sample=dict(lap=0,leg=0,waypoint=1,position=position,native={},
            regional_navigation=dict(route_key=key,points=[end],index=0,mode='coastal',
                policy=REGIONAL_POLICY,probe_position=geo(800,0),job='coastal'))
        navigator=RegionalNavigator(rpc,coastal=True);navigator.coast_index=Near();navigator.key=key
        async def fallback():return dict(route_key=key,points=[geo(6500,5000)],index=0,
            mode='detour',policy=REGIONAL_POLICY,probe_position=position,job='offshore')
        navigator.task=asyncio.create_task(fallback());await navigator.task
        self.assertEqual(navigator.update(sample,geo(6500,9000)),end)
        self.assertEqual(sample['regional_navigation']['job'],'coastal')
        sample['position']=end
        self.assertEqual(navigator.update(sample,geo(6500,9000)),end)
        self.assertEqual(sample['regional_waiting']['reason'],'coastal_renewal')
        await navigator.close()

    async def test_destination_port_survives_a_local_waypoint_target(self):
        from unittest.mock import patch
        async def rpc(path,body=None,timeout=10):
            if path.endswith('/terrain.json'):return grid()
            return dict(id='fixture',status='ready',spec=SPEC,manifest=dict(source='fixture'))
        state=dict(lap=0,leg=0,waypoint=1,position=geo(800,-4000))
        target,port=geo(800,9000),geo(9000,-5000)
        navigator=RegionalNavigator(rpc)
        with patch('posim_routing.coastal.coastal_corridor',return_value=None) as coastal:
            navigator.update(state,target,needed=True,progress_target=port)
            value=await navigator.task
        self.assertEqual(coastal.call_args.kwargs['progress_target'],port)
        self.assertEqual(value['destination_progress_target'],port)
        self.assertEqual(value['mode'],'detour')
        await navigator.close()

    async def test_blocked_coarse_frontier_retries_finer_real_terrain(self):
        from unittest.mock import patch,AsyncMock
        state=dict(lap=0,leg=0,waypoint=1,position=geo(800,-4000))
        async def rpc(path,body=None,timeout=10):
            return grid()
        navigator=RegionalNavigator(rpc)
        navigator.terrain_region=AsyncMock(return_value=dict(id='fixture',spec=SPEC,manifest=dict(source='fixture')))
        target=geo(800,9000)
        with patch('posim_routing.coastal.coastal_corridor',return_value=None), \
             patch('posim_routing.regional.corridor',side_effect=[RuntimeError('regional_corridor_unavailable'),dict(points=[geo(800,1000)],index=0)]):
            plan=await navigator.prepare(state,target,[0,0,1],needed=True)
        self.assertEqual([call.args[1] for call in navigator.terrain_region.call_args_list],[20000,6000])
        self.assertEqual(plan['mode'],'detour')
        self.assertEqual(plan['destination_progress_target'],target)
        await navigator.close()

    async def test_old_coastal_policy_is_replanned_without_moving_the_vehicle(self):
        async def rpc(*args,**kwargs):await asyncio.Event().wait()
        position=geo(800,-4000);key=[0,0,1]
        sample=dict(lap=0,leg=0,waypoint=1,position=position,status='sailing',native={},
            regional_navigation=dict(route_key=key,points=[geo(800,5000)],index=0,mode='coastal',
                policy='buffered-water-v8-mapped-land-exclusion',probe_position=position,job='checked'))
        navigator=RegionalNavigator(rpc)
        navigator.update(sample,geo(800,9000),needed=True)
        self.assertNotIn('regional_navigation',sample)
        self.assertEqual(sample['position'],position)
        self.assertEqual(sample['status'],'sailing')
        self.assertIsNotNone(navigator.task)
        await navigator.close()

    async def test_exhausted_mapped_detour_cannot_fall_back_across_the_peninsula(self):
        async def rpc(*args,**kwargs):await asyncio.Event().wait()
        key=[0,0,1];end=geo(2000,0);target=geo(-8000,-8000)
        sample=dict(lap=0,leg=0,waypoint=1,position=end,status='sailing',native={},
            regional_navigation=dict(route_key=key,points=[end],index=1,mode='detour',
                mapped_guidance=dict(course_target=target),policy=REGIONAL_POLICY,
                probe_position=geo(0,0),job='checked'))
        navigator=RegionalNavigator(rpc);navigator.key=key
        self.assertEqual(navigator.update(sample,target),end)
        self.assertIsNotNone(sample.get('regional_waiting'))
        self.assertIsNotNone(navigator.task)
        await navigator.close()

    async def test_coastal_lookahead_is_prefetched_before_its_end(self):
        async def rpc(*args,**kwargs):
            await asyncio.Event().wait()
        class Near:
            def nearest(self,position):return 1000.,geo(0,0)
        key=[0,0,1];point=geo(800,6000)
        sample=dict(lap=0,leg=0,waypoint=1,position=geo(800,4500),status='sailing',
            regional_navigation=dict(points=[geo(800,3000),point],index=1,mode='coastal',route_key=key,
                                     policy=REGIONAL_POLICY,probe_position=geo(800,0)),native={})
        navigator=RegionalNavigator(rpc,coastal=True);navigator.coast_index=Near()
        target=geo(6500,9000)
        self.assertEqual(navigator.update(sample,target),point)
        pending=navigator.task;self.assertIsNotNone(pending)
        for _ in range(5):
            self.assertEqual(navigator.update(sample,target),point)
            self.assertIs(navigator.task,pending)
        self.assertEqual(sample['status'],'sailing')
        # Exhaust the checked path during a slow download. No tangent may be
        # chained indefinitely, even if it would approach the offshore target.
        sample['position']=geo(800,6000)
        sample['regional_navigation']['job']='checked'
        self.assertEqual(navigator.update(sample,target),point)
        self.assertEqual(sample['regional_waiting']['reason'],'coastal_renewal')
        sample['position']=geo(800,6010)
        self.assertEqual(navigator.update(sample,target),point)
        self.assertNotIn('continuation',sample['regional_navigation'])
        await navigator.close()

    async def test_saved_grid_lookahead_renews_without_discarding_its_checked_target(self):
        async def rpc(*args,**kwargs):await asyncio.Event().wait()
        class Near:
            def nearest(self,position):return 1000.,geo(0,0)
        key=[0,0,1];point=geo(800,9000);position=geo(800,0)
        sample=dict(lap=0,leg=0,waypoint=1,position=position,status='sailing',native={},
            regional_navigation=dict(points=[point],index=0,mode='coastal',route_key=key,
                policy=REGIONAL_POLICY,probe_position=geo(800,-5000),onward_verified=True,
                onward_points=[geo(6500,9000)],job='checked'))
        navigator=RegionalNavigator(rpc,coastal=True);navigator.coast_index=Near()
        self.assertEqual(navigator.update(sample,geo(6500,9000)),point)
        self.assertIsNotNone(navigator.task)
        self.assertEqual(sample['regional_navigation']['job'],'checked')
        self.assertEqual(sample['position'],position)
        self.assertEqual(sample['status'],'sailing')
        await navigator.close()

    async def test_recorded_shape_handoff_preserves_checked_coast(self):
        async def rpc(*args,**kwargs):await asyncio.Event().wait()
        class Near:
            def nearest(self,position):return 4000.,[118.73,24.7]
        old=[0,2,7];point=[118.7441,24.6863]
        sample=dict(lap=0,leg=2,waypoint=8,position=[118.75956451849835,24.67215438376837],
            regional_navigation=dict(route_key=old,points=[point],index=0,mode='coastal',
                                     policy=REGIONAL_POLICY,probe_position=[118.760,24.671],job='checked'),native={})
        navigator=RegionalNavigator(rpc,coastal=True);navigator.coast_index=Near();navigator.key=old
        self.assertEqual(navigator.update(sample,[118.48391605957087,24.503214246085488],retain_coast=True),point)
        self.assertEqual(sample['regional_navigation']['route_key'],[0,2,8])
        self.assertEqual(sample['regional_handoff']['previous_key'],old)
        await navigator.close()

    async def test_handoff_requires_ordinary_shape_in_same_leg_and_lap(self):
        import copy
        async def rpc(*args,**kwargs):await asyncio.Event().wait()
        original=dict(lap=0,leg=2,waypoint=8,position=geo(800,0),
            regional_navigation=dict(route_key=[0,2,7],points=[geo(800,3000)],index=0,mode='coastal',
                                     policy=REGIONAL_POLICY,job='checked'),native={})
        for overrides,retain,terminal in [({},False,False),({'leg':3},True,False),
                    ({'lap':1},True,False),({'waypoint':6},True,False),({},True,True)]:
            sample=copy.deepcopy(original);sample.update(overrides)
            navigator=RegionalNavigator(rpc)
            target=geo(6500,9000) if not terminal else geo(1500,0)
            self.assertEqual(navigator.update(sample,target,retain_coast=retain,terminal=terminal),target)
            self.assertNotIn('regional_handoff',sample)
            await navigator.close()

    async def test_failed_and_empty_renewal_hold_until_a_checked_replacement(self):
        class Near:
            def nearest(self,position):return 1000.,geo(0,0)
        async def rpc(*args,**kwargs):await asyncio.Event().wait()
        key=[0,0,1];end=geo(800,4000);target=geo(6500,9000)
        sample=dict(lap=0,leg=0,waypoint=1,position=end,status='sailing',native={},
            regional_navigation=dict(route_key=key,points=[geo(800,1000),end],index=2,mode='coastal',
                policy=REGIONAL_POLICY,probe_position=geo(800,0),job='checked'))
        navigator=RegionalNavigator(rpc,coastal=True);navigator.coast_index=Near();navigator.key=key
        async def failed():raise RuntimeError('regional_terrain_unavailable')
        navigator.task=asyncio.create_task(failed())
        await asyncio.gather(navigator.task,return_exceptions=True)
        self.assertEqual(navigator.update(sample,target),end)
        self.assertEqual(sample['regional_planning']['status'],'retrying')
        self.assertIsNotNone(sample.get('regional_waiting'))
        for _ in range(10):self.assertEqual(navigator.update(sample,target),end)
        async def empty():return dict(route_key=key,points=[],index=0,mode='course_retained',
            policy=REGIONAL_POLICY,probe_position=end,job='empty')
        navigator.task=asyncio.create_task(empty());await navigator.task
        self.assertEqual(navigator.update(sample,target),end)
        self.assertEqual(sample['regional_navigation']['job'],'checked')
        self.assertIsNotNone(sample.get('regional_waiting'))
        replacement=geo(800,6000)
        async def ready():return dict(route_key=key,points=[replacement],index=0,mode='coastal',
            policy=REGIONAL_POLICY,probe_position=end,job='new')
        navigator.task=asyncio.create_task(ready());await navigator.task
        self.assertEqual(navigator.update(sample,target),replacement)
        self.assertNotIn('regional_waiting',sample)
        self.assertEqual(sample['status'],'sailing')
        await navigator.close()

    async def test_coastal_preference_can_replace_an_active_offshore_detour(self):
        from unittest.mock import patch
        key=[0,0,1];old=geo(6500,1000);coast=geo(800,2000)
        state=dict(lap=0,leg=0,waypoint=1,position=geo(800,-4000),status='sailing',
            regional_navigation=dict(points=[old],index=0,mode='detour',route_key=key,policy=REGIONAL_POLICY),
            native=dict(navigation=dict(route_key=key)))
        class Near:
            def nearest(self,position):return 1000.,geo(0,0)
        async def rpc(path,body=None,timeout=10):
            if path.endswith('/terrain.json'):return grid()
            return dict(id='fixture',status='ready',spec=SPEC,manifest=dict(source='fixture'))
        navigator=RegionalNavigator(rpc,coastal=True);navigator.coast_index=Near()
        with patch('posim_routing.coastal.coastal_corridor',return_value=dict(points=[coast],index=0,mode='coastal')):
            self.assertEqual(navigator.update(state,geo(6500,9000)),old)
            self.assertIsNotNone(navigator.task)
            await navigator.task
            self.assertEqual(navigator.update(state,geo(6500,9000)),coast)
        self.assertNotIn('route_key',state['native']['navigation'])
        self.assertEqual(state['status'],'sailing');await navigator.close()

    async def test_replan_waits_for_warming_coast_index(self):
        from unittest.mock import patch
        class Near:
            def nearest(self,position):return 1000.,geo(0,0)
        async def warm():
            await asyncio.sleep(0);return Near()
        async def rpc(path,body=None,timeout=10):
            if path.endswith('/terrain.json'):return grid()
            return dict(id='fixture',status='ready',spec=SPEC,manifest=dict(source='fixture'))
        navigator=RegionalNavigator(rpc,coastal=True);navigator.index_task=asyncio.create_task(warm())
        state=dict(lap=0,leg=0,waypoint=1,position=geo(800,-4000))
        with patch('posim_routing.coastal.coastal_corridor',return_value=dict(points=[geo(800,2000)],index=0,mode='coastal')):
            value=await navigator.prepare(state,geo(6500,9000),[0,0,1],needed=True)
        self.assertEqual(value['mode'],'coastal');await navigator.close()

    async def test_coast_budget_does_not_disable_required_detour(self):
        from unittest.mock import patch
        state = dict(lap=0, leg=0, waypoint=1, position=geo(800, -4000))
        async def rpc(path, body=None, timeout=10):
            if path.endswith('/terrain.json'):
                return grid()
            return dict(id='fixture', status='ready', spec=SPEC,
                        manifest=dict(source='synthetic fixture'))
        navigator = RegionalNavigator(rpc)
        with patch('posim_routing.coastal.coastal_corridor', side_effect=RuntimeError('coastal_planning_budget_exceeded')), \
             patch('posim_routing.regional.corridor', return_value=dict(points=[geo(3000, 0)], index=0)) as detour:
            value = await navigator.prepare(state, geo(6500, 9000), [0, 0, 1], needed=True)
        self.assertEqual(value['coastal_preference_error'], 'coastal_planning_budget_exceeded')
        self.assertEqual(value['points'], [geo(3000, 0)])
        detour.assert_called_once()
        await navigator.close()

    async def test_completed_coast_plan_replaces_corridor_once_without_pause(self):
        from unittest.mock import patch
        state = dict(lap=0, leg=0, waypoint=1, position=geo(800, -4000), status='sailing',
                     native=dict(navigation=dict(route_key=[0, 0, 1])))
        calls = []
        class Near:
            def nearest(self, position):
                return 1000., geo(0, 0)
        async def rpc(path, body=None, timeout=10):
            calls.append(path)
            if path.endswith('/terrain.json'):
                return grid()
            return dict(id='fixture', status='ready', spec=SPEC,
                        manifest=dict(source='synthetic fixture'))
        navigator = RegionalNavigator(rpc, coastal=True)
        navigator.coast_index = Near()
        target = geo(6500, 9000)
        point = geo(800, 2000)
        with patch('posim_routing.coastal.coastal_corridor', return_value=dict(points=[point], index=0, mode='coastal')):
            self.assertEqual(navigator.update(state, target), state['position'])
            self.assertEqual(state['regional_waiting']['reason'],'land_checked_coastal_route_pending')
            self.assertEqual(state['status'], 'sailing')
            await navigator.task
            self.assertEqual(navigator.update(state, target), point)
        self.assertNotIn('route_key', state['native']['navigation'])
        state['native']['navigation']['route_key'] = [0, 0, 1]
        self.assertEqual(navigator.update(state, target), point)
        self.assertIn('route_key', state['native']['navigation'])
        self.assertEqual(state['status'], 'sailing')
        self.assertTrue(all(path.startswith('/terrain/jobs') for path in calls))
        await navigator.close()

    async def test_ocean_does_not_request_jobs(self):
        async def rpc(*args, **kwargs):
            raise AssertionError('open ocean must not request a coast job')
        navigator = RegionalNavigator(rpc, coastal=True)
        state = dict(lap=0, leg=0, waypoint=1, position=[-150., 0.])
        target = [-145., 0.]
        self.assertEqual(navigator.update(state, target), target)
        await navigator.index_task
        self.assertEqual(navigator.update(state, target), target)
        self.assertEqual(state['coastal_navigation']['status'], 'open_ocean')
        self.assertIsNone(navigator.task)
        await navigator.close()

    async def test_download_uses_fresh_pose_and_result_is_not_reused_on_another_leg(self):
        from unittest.mock import patch
        terrain = grid()
        state = dict(lap=0, leg=0, waypoint=1, position=geo(800, -4000))
        calls, starts = [], []
        async def rpc(path, body=None, timeout=10):
            calls.append(path)
            if path.endswith('/terrain.json'):
                state['position'] = geo(800, -3500)
                return terrain
            return dict(id='terrain-fixture', status='ready', spec=SPEC,
                        manifest=dict(source='synthetic fixture'))
        def plan(terrain, spec, start, target, guard=None, progress_target=None):
            starts.append(start)
            return dict(points=[geo(800, 2000)], index=0, mode='coastal')
        navigator = RegionalNavigator(rpc)
        with patch('posim_routing.coastal.coastal_corridor', plan):
            value = await navigator.prepare(state, geo(6500, 9000), [0, 0, 1], (1000., geo(0, 0)), False)
        self.assertEqual(starts, [state['position']])
        state['regional_navigation'] = value
        state['leg'] = 1
        target = geo(6000, 9000)
        self.assertEqual(navigator.update(state, target), target)
        await navigator.close()

    async def test_unavailable_coast_is_throttled_without_holding_navigation(self):
        async def rpc(*args, **kwargs):
            raise AssertionError('repeated download despite unchanged probe')
        class Near:
            def nearest(self, position):
                return 1000., geo(0, 0)
        navigator = RegionalNavigator(rpc, coastal=True)
        navigator.coast_index = Near()
        navigator.probed = geo(800, -4000)
        state = dict(lap=0, leg=0, waypoint=1, position=geo(800, -3500))
        navigator.key = [0, 0, 1]
        target = geo(6500, 9000)
        self.assertEqual(navigator.update(state, target), target)
        self.assertIsNone(navigator.task)
        await navigator.close()


if __name__ == '__main__':
    unittest.main()
