"""Arrival coverage may need another checked streaming step."""
import math
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock,patch
from posim_routing.port_approach import shortest_arrival,ArrivalOutsideResidentTerrain
from posim_routing.regional import RegionalNavigator,REGIONAL_POLICY

class ArrivalCoverage(unittest.TestCase):
    def test_missing_arrival_coverage_is_distinct_from_shallow_water(self):
        planner=SimpleNamespace(deadline=math.inf,route=lambda *_: self.fail('unknown goal routed'))
        with self.assertRaises(ArrivalOutsideResidentTerrain):
            shortest_arrival(planner,SimpleNamespace(maximum=lambda *_:None),[0,0,0],[[2000,0,0]])
        with self.assertRaisesRegex(RuntimeError,'no_confirmed_corridor'):
            shortest_arrival(planner,SimpleNamespace(maximum=lambda *_:-2.),[0,0,0],[[2000,0,0]])
        blocked=SimpleNamespace(maximum=lambda *_:None,field=SimpleNamespace(maximum=lambda *_:-10.))
        with self.assertRaisesRegex(RuntimeError,'no_confirmed_corridor'):
            shortest_arrival(planner,blocked,[0,0,0],[[2000,0,0]])

class PortRenewal(unittest.IsolatedAsyncioTestCase):
    async def test_course_advance_retains_only_the_same_checked_port(self):
        import copy
        target=[114.161,22.295];point=[114.152,22.300]
        base=dict(lap=0,leg=3,waypoint=10,position=[114.139,22.297],native={},
            regional_navigation=dict(route_key=[0,3,9],points=[point,target],index=0,
                mode='port_approach',arrival_centre=target,arrival_radius_m=300.,policy=REGIONAL_POLICY))
        nav=RegionalNavigator(AsyncMock())
        state=copy.deepcopy(base)
        self.assertEqual(nav.update(state,target,terminal=True),point)
        self.assertEqual(state['regional_navigation']['route_key'],[0,3,10])
        self.assertEqual(state['regional_handoff']['reason'],'same_port_course_advance')
        self.assertIsNone(nav.task)
        for changes,goal in ((dict(leg=4),target),(dict(lap=1),target),({},[114.17,22.295])):
            nav=RegionalNavigator(AsyncMock());state=copy.deepcopy(base);state.update(changes)
            nav.prepare_port=AsyncMock()
            nav.update(state,goal,terminal=True)
            self.assertEqual(state['regional_navigation']['route_key'],[0,3,9])
            self.assertIsNotNone(nav.task)
            await nav.close()

    async def test_unknown_resident_arrival_uses_fine_harbour_rules(self):
        native=dict(session_nonce='world',origin=dict(epoch=4,spec=dict(latitude=22.3,longitude=114.14)),
                    geographic=dict(longitude=114.139,latitude=22.297))
        grid=dict(xs=[-3000,3000],ys=[-3000,3000],heights=[[-10,-10],[-10,-10]])
        async def rpc(path,*args,**kwargs):
            if path=='/promotional/state':return native
            if path=='/stream/state':return dict(nonce='world',tiles=[])
            if path=='/terrain/jobs/region/files/terrain.json':return grid
            self.fail(path)
        nav=RegionalNavigator(rpc);target=[114.161,22.295];key=[0,3,10]
        nav.terrain_region=AsyncMock(return_value=dict(id='region',spec=native['origin']['spec'],manifest=dict(source='real depth')))
        checked=dict(points=[[114.152,22.300],target],index=0,mode='port_approach',arrival_radius_m=300.)
        with patch('posim_routing.land.guard_from_spec',return_value='land'),patch('posim_routing.port_approach.corridor',side_effect=[ArrivalOutsideResidentTerrain(),checked]) as plan:
            value=await nav.prepare_port(dict(native=dict(job='seed')),target,key)
        self.assertEqual(plan.call_args.args[0],[grid])
        self.assertEqual(plan.call_args.kwargs['guard'],'land')
        self.assertEqual(nav.terrain_region.call_args.args[1],6000.)
        self.assertEqual(value['mode'],'port_approach')
        self.assertEqual(value['route_key'],key)
        self.assertEqual(value['source'],'real depth')
        self.assertIn('resident corridor rechecked',value['scope'])

    async def test_verified_runway_is_used_before_resident_arrival_retry(self):
        nav=RegionalNavigator(AsyncMock());key=[0,1,11]
        state=dict(lap=0,leg=1,waypoint=11,position=[118.10,24.425],native={},
            regional_navigation=dict(route_key=key,points=[[118.093,24.428]],index=0,
            mode='port_approach_lookahead',probe_position=[118.10,24.425],policy=REGIONAL_POLICY))
        nav.prepare_port=AsyncMock()
        self.assertEqual(nav.update(state,[118.077,24.436],terminal=True),[118.093,24.428])
        self.assertIsNone(nav.task)
        state['position']=[118.094,24.427]
        nav.update(state,[118.077,24.436],terminal=True)
        self.assertIsNotNone(nav.task)
        await nav.close()

if __name__=='__main__':unittest.main()
