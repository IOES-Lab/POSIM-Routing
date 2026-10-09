"""Port joins must consider offshore overshoots while retaining coast guards."""
import math
import unittest
from types import SimpleNamespace

import shapely
from posim_routing.connections import CoastConnector, shorten, length


class Connections(unittest.TestCase):
    def test_300km_join_removes_offshore_overshoot(self):
        points=[[118.04573482228693,24.16856318321221], [117.,23.],
                [114.1,21.7], [114.16099724240206,22.294787745002328]]
        calls=[]
        def route(start,end):
            calls.append(start)
            return [start,end]
        result,_=shorten(points,SimpleNamespace(route=route))
        self.assertIn(points[1],calls)
        self.assertNotIn(points[2],result)
        self.assertLess(length(result),length(points)-50000.)

    def test_long_connection_grid_keeps_continuous_coast_margin(self):
        connector=object.__new__(CoastConnector)
        connector.step=250.;connector.margin=80.;connector.timeout=15.
        land=shapely.box(289000.,95500.,291000.,98000.)
        points=connector.plan(land,(0.,0.),(300000.,100000.))
        self.assertGreater(len(points),2)
        for a,b in zip(points,points[1:]):
            self.assertFalse(land.buffer(80.).intersects(shapely.LineString([a,b])))
        self.assertLess(sum(math.dist(a,b) for a,b in zip(points,points[1:])),325000.)

    def test_resource_limit_and_preserved_waypoint_are_not_bypassed(self):
        connector=object.__new__(CoastConnector)
        connector.step=250.;connector.margin=80.;connector.timeout=5.
        with self.assertRaisesRegex(RuntimeError,'connection_grid_limit'):
            connector.plan(shapely.box(149000.,149000.,151000.,151000.),(0.,0.),(300000.,300000.))
        points=[[117.,23.],[114.1,21.7],[114.16099724240206,22.294787745002328]]
        result,_=shorten(points,SimpleNamespace(route=lambda a,b:[a,b]),protected=[points[1]])
        self.assertEqual(result,points)
        result,_=shorten(points,SimpleNamespace(route=lambda a,b:[a,b]),minimum_index=len(points))
        self.assertEqual(result,points)


if __name__=='__main__':unittest.main()
