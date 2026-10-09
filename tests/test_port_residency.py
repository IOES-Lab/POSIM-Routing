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
