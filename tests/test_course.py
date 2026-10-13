"""Measured shape progress must not require offshore visits or skip arrivals."""
import copy
import unittest
from posim_routing.course import course_target,passed_coastal_shape

LEG=dict(coordinates=[[119.761963,25.085599],[118.937988,24.542126],
                      [118.48391605957087,24.503214246085488]],canal=False)

def state(position):
    return dict(lap=0,leg=2,waypoint=1,position=position,
                coastal_navigation=dict(status='coastal_region'))

class CourseTests(unittest.TestCase):
    def test_distant_offshore_shape_rounds_only_with_checked_matching_onward_goal(self):
        leg=dict(coordinates=[[109.,16.1],[110.1,12.2],[107.3207,7.6091]],canal=False)
        sample=state([109.2413,14.4598])
        self.assertEqual(course_target(leg,1,sample),leg['coordinates'][1])
        sample['regional_navigation']=dict(mode='coastal',route_key=[0,2,1],
                                          coastal_course_target=leg['coordinates'][2])
        self.assertEqual(course_target(leg,1,sample),leg['coordinates'][2])
        sample['course_lookahead']=dict(route_key=[0,2,1],target=leg['coordinates'][2],rounded_shape=True)
        sample['position']=[109.1,11.8]
        self.assertTrue(passed_coastal_shape(leg,1,sample))
        sample.pop('course_lookahead');sample['regional_navigation']['route_key']=[0,1,1]
        sample['position']=[109.2413,14.4598]
        self.assertEqual(course_target(leg,1,sample),leg['coordinates'][1])

    def test_first_da_nang_offshore_shape_passes_on_measured_onward_progress(self):
        leg=dict(coordinates=[[108.39851611053206,16.129448766936086],
                              [109.,16.1],[110.1,12.2]],canal=False,
            fixed_departure=dict(policy='fixed-port-waypoints-v1',verified=True,
                                 first_waypoint=0,last_waypoint=0))
        sample=state([108.4,15.8]);goal=course_target(leg,1,sample)
        self.assertEqual(goal,leg['coordinates'][2])
        # No accepted rounding or no outgoing-plane passage cannot skip it.
        self.assertFalse(passed_coastal_shape(leg,1,sample))
        sample['course_lookahead']=dict(route_key=[0,2,1],target=goal,rounded_shape=True)
        self.assertTrue(passed_coastal_shape(leg,1,sample))
        sample['position']=[108.4,16.13]
        self.assertFalse(passed_coastal_shape(leg,1,sample))
        sample['position']=[108.4,15.8];leg['passage_gates']=[leg['coordinates'][1]]
        self.assertFalse(passed_coastal_shape(leg,1,sample))

    def test_recorded_haikou_rounding_does_not_reverse_at_eighty_kilometres(self):
        leg=dict(coordinates=[[113.92911887365759,22.113265931524595],
                              [114.10000000000002,21.7],[111.35000000000002,20.4]],canal=False)
        sample=state([113.33915848352626,21.62383578337067])
        goal=course_target(leg,1,sample)
        self.assertEqual(goal,leg['coordinates'][2])
        sample['course_lookahead']=dict(route_key=[0,2,1],target=goal,rounded_shape=True)
        # Recorded zigzag positions straddle the discovery/rounding boundary.
        for position in ([113.32801335937972,21.622695168329194],
                         [113.30496710768507,21.74300540396193],[112.5,21.1]):
            sample['position']=position
            sample['coastal_navigation']['status']='open_ocean'
            self.assertEqual(course_target(leg,1,sample),goal)
        self.assertTrue(passed_coastal_shape(leg,1,sample))
        sample['course_lookahead']['route_key']=[0,1,1]
        self.assertEqual(course_target(leg,1,sample),leg['coordinates'][1])
        self.assertFalse(passed_coastal_shape(leg,1,sample))

    def test_rounded_goal_cannot_override_a_fixed_bend_or_passage(self):
        leg=copy.deepcopy(LEG);sample=state([118.85,24.8])
        sample['course_lookahead']=dict(route_key=[0,2,1],target=leg['coordinates'][2],rounded_shape=True)
        leg['fixed_departure']=dict(policy='fixed-port-waypoints-v1',verified=True,first_waypoint=0,last_waypoint=1)
        self.assertEqual(course_target(leg,1,sample),leg['coordinates'][1])
        self.assertFalse(passed_coastal_shape(leg,1,sample))
        leg.pop('fixed_departure');leg['passage_gates']=[leg['coordinates'][1]]
        self.assertEqual(course_target(leg,1,sample),leg['coordinates'][1])
        self.assertFalse(passed_coastal_shape(leg,1,sample))

    def test_recorded_shenhu_shape_uses_onward_course(self):
        self.assertEqual(course_target(LEG,1,state([118.85,24.8])),LEG['coordinates'][2])
        self.assertTrue(passed_coastal_shape(LEG,1,state([118.70,24.65])))
        self.assertFalse(passed_coastal_shape(LEG,1,state([117.5,24.65])))

    def test_required_passage_and_port_stay_strict(self):
        gate=copy.deepcopy(LEG);gate['passage_gates']=[LEG['coordinates'][1]]
        self.assertEqual(course_target(gate,1,state([118.85,24.8])),LEG['coordinates'][1])
        self.assertFalse(passed_coastal_shape(gate,1,state([118.70,24.65])))
        canal=copy.deepcopy(LEG);canal['canal']=True
        self.assertEqual(course_target(canal,1,state([118.85,24.8])),LEG['coordinates'][1])
        self.assertFalse(passed_coastal_shape(LEG,2,state(LEG['coordinates'][2])))
        self.assertEqual(course_target(LEG,2,state([118.70,24.65])),LEG['coordinates'][2])

    def test_ocean_distant_shapes_and_hairpins_keep_course(self):
        ocean=state([118.85,24.8]);ocean.pop('coastal_navigation')
        self.assertEqual(course_target(LEG,1,ocean),LEG['coordinates'][1])
        self.assertEqual(course_target(LEG,1,state([117.5,24.65])),LEG['coordinates'][1])
        hairpin=dict(coordinates=[[118.8,24.5],[118.9,24.5],[118.81,24.5]])
        self.assertEqual(course_target(hairpin,1,state([118.88,24.5])),hairpin['coordinates'][1])

if __name__=='__main__':unittest.main()
