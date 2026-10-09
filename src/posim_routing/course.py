"""Advance coarse course vertices using measured coastal progress, never ports.

The maritime graph supplies a route shape, not a sequence of mandatory offshore
visits. Native terrain checks still approve every actual motion corridor.
"""
import math
from pyproj import Geod

GEO = Geod(ellps='WGS84')


def passed_vertex(previous, vertex, following, position, maximum_offset=80000., minimum_leg=2000., corner_margin=2000.):
    incoming_back, _, incoming = GEO.inv(*vertex, *previous)
    outgoing_bearing, _, outgoing = GEO.inv(*vertex, *following)
    bearing, _, separation = GEO.inv(*vertex, *position)
    if min(incoming, outgoing) < minimum_leg:
        return False
    incoming_bearing = incoming_back + 180.
    # A hairpin, short port approach or distant parallel route is not evidence
    # that its vertex has been passed. Only round ordinary progressive bends.
    if math.cos(math.radians(outgoing_bearing-incoming_bearing)) < -.25:
        return False
    if separation > min(maximum_offset, min(incoming, outgoing)*.5):
        return False
    along_in = separation*math.cos(math.radians(bearing-incoming_bearing))
    along_out = separation*math.cos(math.radians(bearing-outgoing_bearing))
    return along_in >= 0 and along_out >= -min(corner_margin, min(incoming, outgoing)*.1)


def coastal_shape(leg, index, state):
    points = leg['coordinates']
    if not 0 < index < len(points)-1 or leg.get('canal'):
        return False
    plan = state.get('regional_navigation') or {}
    regional=(plan.get('mode') in ('coastal', 'coastal_approach')
              and plan.get('route_key') == [state['lap'], state['leg'], index])
    if not regional and (state.get('coastal_navigation') or {}).get('status') != 'coastal_region':
        return False
    # Keep mapped canal samples and explicit southern-ocean gates mandatory.
    from posim_terrain.canal_override import in_region
    if any(in_region(p) for p in (state['position'], *points[index-1:index+2])):
        return False
    if any(GEO.inv(*points[index], *gate)[2] < 50 for gate in leg.get('passage_gates', [])):
        return False
    return True


def passed_coastal_shape(leg, index, state):
    points=leg['coordinates']
    return coastal_shape(leg,index,state) and passed_vertex(
        points[index-1],points[index],points[index+1],state['position'])


def course_target(leg,index,state):
    """Round an offshore shape vertex before it pulls a coastal path to sea."""
    points=leg['coordinates'];vertex=points[index]
    if not coastal_shape(leg,index,state):return vertex
    incoming_back,_,incoming=GEO.inv(*vertex,*points[index-1])
    bearing,_,outgoing=GEO.inv(*vertex,*points[index+1])
    if min(incoming,outgoing)<2000 or math.cos(math.radians(bearing-incoming_back-180.))<-.25:return vertex
    if GEO.inv(*state['position'],*vertex)[2]>min(80000.,min(incoming,outgoing)*.5):return vertex
    return list(GEO.fwd(*vertex,bearing,min(25000.,outgoing*.25))[:2])
