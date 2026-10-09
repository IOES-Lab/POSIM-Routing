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
