"""Mandatory bends, arrival eligibility and unknown resident terrain holds."""
import hashlib,json,math,unittest
from unittest.mock import AsyncMock
import numpy as np
from posim_routing.port_waypoints import POLICY,BUFFER_M,DEPTH_M,split_incoming,mandatory,verify_book,native_points,resident_fixed_route,offline_planner
from posim_routing.course import course_target,passed_coastal_shape
from posim_routing.navigation import TerrainField,DetourPlanner
from posim_routing.planning import CachedTerrain
from posim_routing.regional import RegionalNavigator


def leg():
    points=[[114.13,22.29],[114.14,22.29],[114.14,22.30],[114.16,22.30]]
    value=dict(policy=POLICY,verified=True,first_waypoint=1,buffer_m=BUFFER_M,
        required_depth_m=DEPTH_M,minimum_depth_m=3.,waypoint_radius_m=5.,source=['fixture'],
        coordinates_sha256=hashlib.sha256(json.dumps(points[1:],separators=(',',':')).encode()).hexdigest())
    return dict(coordinates=points,canal=False,fixed_approach=value)


class FixedCourse(unittest.TestCase):
    def test_fixed_harbour_bend_is_never_rounded_or_passed_offshore(self):
        route=leg();state=dict(lap=0,leg=3,waypoint=1,position=[114.139,22.299],coastal_navigation=dict(status='coastal_region'))
        self.assertEqual(course_target(route,1,state),route['coordinates'][1])
        self.assertFalse(passed_coastal_shape(route,1,state))
        verify_book(dict(legs=[route]))
        route['coordinates'][2][0]+=.001
        with self.assertRaisesRegex(ValueError,'checksum'):verify_book(dict(legs=[route]))

    def test_incomplete_clearance_certificate_is_rejected(self):
        route=leg();route['fixed_approach'].pop('buffer_m')
        with self.assertRaisesRegex(ValueError,'clearance'):verify_book(dict(legs=[route]))

    def test_global_split_is_on_the_incoming_polyline(self):
        prefix,tail=split_incoming([[114.,22.],[114.1,22.],[114.2,22.]],5000.)
        self.assertEqual(prefix[-1],tail[0]);self.assertEqual(tail[-1],[114.2,22.])
        from posim_routing.port_waypoints import GEO
        self.assertAlmostEqual(GEO.inv(*tail[0],*tail[-1])[2],5000.,places=5)

    def test_native_prefix_retains_turn_before_its_bounded_endpoint(self):
        route=leg();points=native_points(route,2,[114.14,22.29001],maximum=1600.)
        self.assertEqual(points[0],route['coordinates'][2])
        self.assertGreater(points[-1][0],114.14)
        self.assertLess(points[-1][0],114.16)
        self.assertEqual(native_points(route,1,[114.139,22.29]),[])

    def test_unknown_fixed_endpoint_never_becomes_a_water_frontier(self):
        axis=np.linspace(-1000,1000,129);heights=np.full((129,129),-10.)
        heights[:,70:]=np.nan
        planner=DetourPlanner(TerrainField(axis,axis,heights),[0.,0.,0.],True)
        with self.assertRaisesRegex(RuntimeError,'fixed_waypoint_corridor_pending'):
            resident_fixed_route(planner,[0.,0.,0.],[[200.,0.,0.]])
        self.assertEqual(resident_fixed_route(planner,[0.,0.,0.],[[-100.,0.,0.],[-100.,200.,0.]]),[[-100.,0.,0.],[-100.,200.,0.]])

    def test_search_does_not_cross_unknown_cells_or_shallow_land(self):
        axis=np.linspace(-1200,1200,257);xx,yy=np.meshgrid(axis,axis)
        heights=np.where((abs(xx)<200)&(yy<450),5.,-10.)
        field=CachedTerrain(TerrainField(axis,axis,heights))
        planner=offline_planner(field,[-800.,0.,0.],step=16.,timeout=30.)
        path=planner.route([-800.,0.,0.],[800.,0.,0.])
        self.assertGreater(max(p[1] for p in path),450.)
        self.assertTrue(all(planner.segment(a,b) for a,b in zip([[-800.,0.,0.],*path[:-1]],path)))

    def test_diagonal_channel_checks_edges_before_accepting_a_turn(self):
        axis=np.arange(-80.,281.,2.);xx,yy=np.meshgrid(axis,axis)
        water=(np.minimum(abs(xx-yy),abs(xx+yy-192.))<28.)|((xx-96.)**2+(yy-96.)**2<45.**2)
        field=CachedTerrain(TerrainField(axis,axis,np.where(water,-10.,5.)))
        start=[0.,0.,0.];goal=[192.,0.,0.]
        planner=offline_planner(field,start,step=16.,timeout=30.)
        self.assertFalse(planner.segment(start,goal))
        path=planner.route(start,goal)
        self.assertGreater(max(p[1] for p in path),60.)
        self.assertTrue(all(planner.segment(a,b) for a,b in zip([start,*path[:-1]],path)))


class FixedRegional(unittest.IsolatedAsyncioTestCase):
    async def test_fixed_approach_cancels_scenic_replan_and_keeps_measured_index(self):
        nav=RegionalNavigator(AsyncMock());route=leg()
        state=dict(lap=0,leg=3,waypoint=2,position=[114.14,22.295],native=dict(job='resident'))
        point=nav.update(state,route['coordinates'][-1],terminal=True,fixed=route)
        self.assertEqual(point,route['coordinates'][2])
        self.assertEqual(state['regional_navigation']['index'],1)
        self.assertTrue(state['regional_navigation']['fixed'])
        self.assertIsNone(nav.task)

class HarbourDeparture(unittest.IsolatedAsyncioTestCase):
    async def test_departure_discards_previous_arrival_and_checks_resident_exit(self):
        from unittest.mock import patch
        import shapely
        from posim_routing.land import LandGuard
        spec=dict(latitude=22.3,longitude=114.16,width_m=4000.,height_m=4000.,samples=129)
        position=[114.16,22.3];axis=np.linspace(-2000,2000,129)
        native=dict(session_nonce='same',origin=dict(epoch=4,spec=spec),geographic=dict(longitude=position[0],latitude=position[1]))
        grid=dict(xs=axis.tolist(),ys=axis.tolist(),heights=np.full((129,129),-10.).tolist())
        async def rpc(path,**kwargs):
            if path=='/promotional/state':return native
            if path=='/stream/state':return dict(nonce='same',tiles=[dict(key='resident')])
            return grid
        nav=RegionalNavigator(rpc,coastal=True)
        state=dict(lap=0,leg=4,waypoint=1,position=position,native=dict(job='same'),
            last_stopover=dict(departed_at=1.,port=dict(longitude=position[0],latitude=position[1])),
            regional_navigation=dict(route_key=[0,3,10],points=[position],index=0,mode='port_approach'))
        target=[114.12,22.29]
        guard=LandGuard(shapely.GeometryCollection(),shapely.box(-64000,-64000,64000,64000))
        with patch('posim_routing.land.guard_from_spec',return_value=guard):
            self.assertEqual(nav.update(state,target),position)
            await nav.task
            goal=nav.update(state,target)
        self.assertEqual(state['regional_navigation']['mode'],'port_departure')
        self.assertEqual(state['regional_navigation']['route_key'],[0,4,1])
        self.assertLess(math.dist(goal,target),math.dist(position,target))
        from posim_routing.port_waypoints import GEO
        self.assertAlmostEqual(GEO.inv(*position,*goal)[2],1000.,places=3)
        self.assertFalse(state.get('regional_waiting'))
        await nav.close()

if __name__=='__main__':unittest.main()
