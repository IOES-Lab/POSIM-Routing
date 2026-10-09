"""Coarse maritime graph routes with optional explicit passage gates."""
import math
from pyproj import Geod

def coordinate(value):
    if len(value)!=2 or not all(isinstance(x,(int,float)) and math.isfinite(x) for x in value):
        raise ValueError('invalid_geographic_coordinate')
    if not -180<=value[0]<=180 or not -90<=value[1]<=90:
        raise ValueError('invalid_geographic_coordinate')
    return list(value)

def route(start,end,*,restrictions=('northwest','panama'),passage_gates=()):
    import searoute
    endpoints=[coordinate(start),*(coordinate(p) for p in passage_gates),coordinate(end)]
    coordinates=[];length=0.
    for a,b in zip(endpoints,endpoints[1:]):
        result=searoute.searoute(a,b,units='m',append_orig_dest=True,restrictions=list(restrictions))
        points=[[(p[0]+180)%360-180,p[1]] for p in result['geometry']['coordinates']]
        coordinates.extend(points if not coordinates else points[1:])
        length+=result['properties']['length']
    return {'type':'Feature','geometry':{'type':'LineString','coordinates':coordinates},
            'properties':{'length_m':length,'passage_gates':list(passage_gates),
                          'restrictions':list(restrictions),'collision_checked':False,
                          'source':'searoute 1.4.3 / Eurostat maritime graph'}}
