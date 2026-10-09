"""Coastal route contracts. Synthetic fixtures are not voyage evidence."""
import asyncio
import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
from posim_routing.coastal import CoastIndex, coastal_corridor, coastal_approach, MINIMUM_DEPTH_M, REGIONAL_BUFFER_M, STANDOFF_M
from posim_routing.navigation import TerrainField, DetourPlanner
from posim_routing.planning import CachedTerrain
from posim_routing.regional import RegionalNavigator,REGIONAL_POLICY,distance
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

    def test_stays_near_coast_and_progresses(self):
        terrain = grid()
        start, target = geo(800, -4000), geo(6500, 9000)
        plan = coastal_corridor(terrain, SPEC, start, target)
        self.assertIsNotNone(plan)
        self.assertEqual(plan['mode'], 'coastal')
        self.assertLess(abs(plan['endpoint_standoff_m']-STANDOFF_M), 200)
        self.assertGreater(plan['progress_to_course_waypoint_m'], 1000)
        self.assert_swept_safe(terrain, start, plan)

    def test_close_safe_start_is_not_rejected_by_an_extra_grid_ring(self):
        terrain = grid()
        start = geo(250, -4000)
        plan = coastal_corridor(terrain, SPEC, start, geo(6500, 9000))
        self.assertIsNotNone(plan)
        self.assertLess(plan['endpoint_standoff_m'], 350.)
        self.assert_swept_safe(terrain, start, plan)

    def test_offshore_start_approaches_visible_coast(self):
        terrain = grid()
        start = geo(5000, -4000)
        plan = coastal_corridor(terrain, SPEC, start, geo(6500, 9000))
        self.assertIsNotNone(plan)
        self.assertLessEqual(plan['endpoint_standoff_m'], 800.)
        self.assertGreater(plan['progress_to_course_waypoint_m'], 1000.)
        self.assert_swept_safe(terrain, start, plan)

    def test_headland_cannot_be_shortcut(self):
        terrain = grid(headland=True)
        start = geo(800, -4000)
        plan = coastal_corridor(terrain, SPEC, start, geo(6500, 9000))
        self.assertIsNotNone(plan)
        points = np.array([FORWARD.transform(*p) for p in plan['points']])
        self.assertGreater(points[:, 0].max(), 2000)
        self.assert_swept_safe(terrain, start, plan)

    def test_shallow_and_unknown_coast_is_not_a_destination(self):
        terrain = grid()
        heights = np.array(terrain['heights'])
        x = np.array(terrain['xs'])
        heights[:, (x > 0) & (x < 1700)] = -1.
        terrain['heights'] = heights.tolist()
        self.assertIsNone(coastal_corridor(terrain, SPEC, geo(3000, -4000), geo(6500, 9000)))
        heights[:, (x > 0) & (x < 1700)] = np.nan
        terrain['heights'] = heights.tolist()
        self.assertIsNone(coastal_corridor(terrain, SPEC, geo(3000, -4000), geo(6500, 9000)))

    def test_two_point_five_metre_coast_is_available_to_wamv(self):
        terrain=grid();heights=np.array(terrain['heights'])
        heights[heights<0]=-2.5;terrain['heights']=heights.tolist()
        start=geo(800,-4000);plan=coastal_corridor(terrain,SPEC,start,geo(6500,9000))
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

    def test_real_discovery_index_wraps_longitude(self):
        index = CoastIndex()
        distance, _ = index.nearest([129.08468, 35.07446])
        self.assertLess(distance, 5000)
        a = index.nearest([179.9, -16.5])
        b = index.nearest([-180.1, -16.5])
        self.assertAlmostEqual(a[0], b[0], places=6)

    def test_distant_coast_approach_advances_without_changing_course_target(self):
        from posim_routing.regional import distance
        start, target, shore = geo(30000, 0), geo(0, -90000), geo(0, -12000)
        candidate=coastal_approach(start,target,shore)
        self.assertIsNotNone(candidate)
        self.assertLessEqual(distance(start,candidate),7500.01)
        self.assertGreater(distance(start,target)-distance(candidate,target),500)
        self.assertGreater(distance(start,shore)-distance(candidate,shore),500)
        self.assertEqual(target,geo(0,-90000))
        self.assertIsNone(coastal_approach(geo(800,0),target,geo(0,0)))
        self.assertIsNone(coastal_approach(geo(30000,0),geo(90000,0),geo(0,0)))

    def test_real_progressive_shore_excludes_backward_coast(self):
        from posim_routing.regional import distance
        index=CoastIndex();start=[122.72,29.85];target=[118.0835,24.4363]
        nearest=index.progressive_shore(start,target)
        self.assertIsNotNone(nearest)
        self.assertLess(nearest[0],80000)
        self.assertGreater(distance(start,target)-distance(nearest[1],target),999)


class BackgroundPlanning(unittest.IsolatedAsyncioTestCase):
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
        # If slow background preparation outlasts the old coast plan, propose
        # its tangent, not the offshore graph point. Goal stays fixed between
        # samples, and native collision approval remains required downstream.
        sample['position']=geo(800,6100)
        continuation=navigator.update(sample,target)
        self.assertLess(distance(sample['position'],continuation),751)
        self.assertNotEqual(continuation,target)
        sample['position']=geo(800,6110)
        self.assertEqual(navigator.update(sample,target),continuation)
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
            self.assertEqual(navigator.update(state, target), target)
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
        def plan(terrain, spec, start, target):
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
