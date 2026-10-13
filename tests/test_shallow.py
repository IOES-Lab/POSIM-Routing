"""A shallow shelf is a route boundary; pre-existing proximity exits outward."""
import unittest
import numpy as np
from posim_routing.shallow import exit_corridor,clearance_field,increasing_clearance
from posim_routing.navigation import TerrainField,DetourPlanner
from posim_routing.planning import CachedTerrain
from posim_terrain.terrain import projection

SPEC=dict(latitude=35.,longitude=129.,width_m=6000,height_m=6000,samples=257)
_,F,I=projection(SPEC)
def geo(x,y):return list(I.transform(x,y))
def terrain():
    axis=np.linspace(-3000,3000,257);x,y=np.meshgrid(axis,axis)
    return dict(xs=axis.tolist(),ys=axis.tolist(),heights=np.where(x<0,-1.,-15.).tolist())

class ShallowExit(unittest.TestCase):
    def test_already_clear_position_needs_no_departure(self):
        self.assertIsNone(exit_corridor(terrain(),SPEC,geo(400,0),geo(1000,2000)))
    def test_inside_buffer_exits_under_propulsion_with_increasing_separation(self):
        grid=terrain();start=geo(100,0)
        value=exit_corridor(grid,SPEC,start,geo(1000,2000))
        self.assertEqual(value['mode'],'shallow_exit');self.assertEqual(value['shallow_water_buffer_m'],200.)
        raw=TerrainField(grid['xs'],grid['ys'],grid['heights']);a=[*F.transform(*start),0.];b=[*F.transform(*value['points'][0]),0.]
        planner=DetourPlanner(CachedTerrain(raw),a,True,maximum_radius=2000)
        planner.radius=200.;planner.clearance=2.
        self.assertTrue(planner.safe(b[:2],0.))
        planner.radius=40.;self.assertTrue(planner.segment(a,b))
        self.assertTrue(increasing_clearance(clearance_field(raw),[a,b]))
    def test_departure_rejects_shallow_hull_and_a_return_towards_the_shelf(self):
        grid=terrain();raw=TerrainField(grid['xs'],grid['ys'],grid['heights'])
        with self.assertRaisesRegex(RuntimeError,'shallow_exit_hull_unsafe'):
            exit_corridor(grid,SPEC,geo(-100,0),geo(1000,2000))
        self.assertFalse(increasing_clearance(clearance_field(raw),[[100,0,0],[300,0,0],[100,0,0]]))
if __name__=='__main__':unittest.main()
