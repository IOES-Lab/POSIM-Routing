"""Negative bathymetry cannot override mapped land. Synthetic regressions."""
import unittest
import math
import tempfile
import zipfile
from pathlib import Path
from unittest.mock import patch

import numpy as np
import shapely

from posim_routing.land import LandGuard, LandExcludedField, guard_from_spec
from posim_routing.navigation import TerrainField, DetourPlanner
from posim_routing.planning import CachedTerrain
from posim_routing.coastal import coastal_corridor
from posim_routing.land_guidance import detour,local_target
from posim_terrain.terrain import projection

SPEC=dict(latitude=35.,longitude=129.,width_m=20000.,height_m=20000.,samples=257)
_,FORWARD,INVERSE=projection(SPEC)


class LandExclusion(unittest.TestCase):
    def test_deep_numeric_cache_cannot_admit_reclaimed_land(self):
        axis=np.arange(-1000.,1001.,50.)
        numeric=TerrainField(axis,axis,np.full((len(axis),len(axis)),-20.))
        guard=LandGuard(shapely.box(-200,-300,200,300),shapely.box(-1100,-1100,1100,1100))
        field=LandExcludedField(CachedTerrain(numeric),guard)
        self.assertIsNone(field.maximum(0,0,8.))
        self.assertEqual(field.maximum(-800,0,8.),-20.)
        planner=DetourPlanner(field,[-800,0,0],True,step=50,timeout=5,maximum_radius=2000)
        planner.radius=8.
        route=planner.route([-800,0,0],[800,0,0])
        self.assertGreater(max(abs(p[1]) for p in route),300.)
        self.assertTrue(all(guard.segment(a,b,8.) for a,b in zip([[-800,0,0],*route[:-1]],route)))
        np.testing.assert_array_equal(numeric.heights,-20.)

    def test_thin_barrier_between_grid_vertices_is_blocked(self):
        axis=np.arange(-200.,201.,100.)
        guard=LandGuard(shapely.box(44,-180,46,180),shapely.box(-1000,-1000,1000,1000))
        self.assertFalse(guard.segment([-100,0],[100,0],8.))
        self.assertFalse(guard.grid(axis,axis,80.)[2,2])
        self.assertTrue(guard.segment([-100,-300],[100,-300],8.))

    def test_land_extends_coastal_search_when_numeric_field_calls_it_water(self):
        axis=np.linspace(-10000.,10000.,257);xx,yy=np.meshgrid(axis,axis)
        numeric=np.where(xx<=0,20.,-20.)
        grid=dict(xs=axis.tolist(),ys=axis.tolist(),heights=numeric.tolist())
        land=shapely.union_all([shapely.box(-11000,-11000,0,11000),shapely.box(0,-1000,1800,1500)])
        guard=LandGuard(land,shapely.box(-11000,-11000,11000,11000))
        start=list(INVERSE.transform(800,-4000));target=list(INVERSE.transform(6500,9000))
        plan=coastal_corridor(grid,SPEC,start,target,guard=guard)
        self.assertIsNotNone(plan)
        points=[[*FORWARD.transform(*p),0] for p in [start,*plan['points'],*plan['onward_points']]]
        self.assertGreater(max(p[0] for p in points),1800.)
        self.assertTrue(all(guard.segment(a,b,40.) for a,b in zip(points,points[1:])))

    def test_unknown_coverage_and_missing_dataset_are_rejected(self):
        guard=LandGuard(shapely.GeometryCollection(),shapely.box(-100,-100,100,100))
        self.assertFalse(guard.segment([0,0],[200,0],8.))
        with patch.dict('os.environ',{'WWOS_COAST_OSM_ARCHIVE':'/nonexistent/land.zip'}):
            with self.assertRaisesRegex(RuntimeError,'data_unavailable'):
                guard_from_spec(SPEC)

    def test_shortened_onward_segments_keep_a_required_mapped_land_turn(self):
        axis=np.linspace(-10000.,10000.,257);xx,yy=np.meshgrid(axis,axis)
        grid=dict(xs=axis.tolist(),ys=axis.tolist(),heights=np.where(xx<=0,20.,-20.).tolist())
        # Both numeric terrain and unbuffered grid vertices miss this thin bank.
        land=shapely.union_all([shapely.box(-11000,-11000,0,11000),
                               shapely.box(3000,8000,3002,9500)])
        guard=LandGuard(land,shapely.box(-11000,-11000,11000,11000))
        start=list(INVERSE.transform(800,-4000));target=list(INVERSE.transform(6500,9000))
        plan=coastal_corridor(grid,SPEC,start,target,guard=guard)
        self.assertIsNotNone(plan)
        onward=[[*FORWARD.transform(*p),0.] for p in [plan['points'][-1],*plan['onward_points']]]
        self.assertFalse(guard.segment(onward[0],onward[-1],40.))
        self.assertGreater(len(plan['onward_points']),1)
        self.assertTrue(all(guard.segment(a,b,40.) for a,b in zip(onward,onward[1:])))
        np.testing.assert_allclose(plan['onward_points'][-1],plan['onward_exit'],atol=1e-10)

    def test_nearer_disconnected_exits_do_not_hide_reachable_water_exit(self):
        axis=np.linspace(-10000.,10000.,257)
        grid=dict(xs=axis.tolist(),ys=axis.tolist(),heights=np.full((257,257),-20.).tolist())
        guard=LandGuard(shapely.box(-200,-11000,0,11000),shapely.box(-11000,-11000,11000,11000))
        start=list(INVERSE.transform(800,0));target=list(INVERSE.transform(-20000,-20000))
        plan=coastal_corridor(grid,SPEC,start,target,guard=guard)
        self.assertIsNotNone(plan)
        exit_x,exit_y=FORWARD.transform(*plan['onward_exit'])
        self.assertGreater(exit_x,0.)
        self.assertLess(exit_y,-9000.)

    def test_complete_water_detour_can_require_initial_distance_loss(self):
        # A bay opens east, while the onward course is west of its banks.
        land=shapely.union_all([shapely.box(-2500,-6000,-1500,6000),
                               shapely.box(-2500,5000,4000,6000),
                               shapely.box(-2500,-6000,4000,-5000)])
        guard=LandGuard(land,shapely.box(-30000,-30000,30000,30000))
        start=[0,0];goal=[-8000,-8000]
        points=detour(guard,start,goal)
        self.assertGreater(points[0][0],4000.)
        self.assertGreater(math.dist(points[0],goal),math.dist(start,goal))
        self.assertEqual(points[-1],goal)
        self.assertTrue(all(guard.segment(a,b,40.) for a,b in zip([start,*points[:-1]],points)))
        local=local_target(start,points,1000.)
        self.assertAlmostEqual(math.dist(start,local),1000.)
        self.assertTrue(guard.segment(start,local,40.))

    def test_mapped_guidance_cannot_place_a_goal_on_land(self):
        guard=LandGuard(shapely.box(-1000,-1000,1000,1000),shapely.box(-30000,-30000,30000,30000))
        with self.assertRaisesRegex(RuntimeError,'endpoint_blocked'):
            detour(guard,[-5000,0],[0,0])

    def test_land_dataset_queries_both_sides_of_date_line(self):
        spec={**SPEC,'longitude':179.999}
        _,forward,inverse=projection(spec)
        west=shapely.box(179.997,34.997,180.,35.003)
        east=shapely.box(-180.,34.997,-179.997,35.003)
        def read(*args,bbox,**kwargs):
            geometry=west if bbox[0]>0 else east
            return {'crs':'EPSG:4326'},None,np.array([geometry.wkb],dtype=object),None
        with tempfile.TemporaryDirectory() as folder:
            archive=Path(folder)/'land.zip'
            with zipfile.ZipFile(archive,'w') as out:out.writestr('land.shp',b'fixture')
            with patch.dict('os.environ',{'WWOS_COAST_OSM_ARCHIVE':str(archive)}),patch('pyogrio.raw.read',side_effect=read) as loader:
                guard=guard_from_spec(spec)
                self.assertEqual(loader.call_count,2)
                self.assertFalse(guard.segment([-1000,0],[1000,0],8.))
                self.assertTrue(guard.segment([-1000,1000],[1000,1000],8.))


if __name__=='__main__':unittest.main()
