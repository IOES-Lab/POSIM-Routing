"""Private pilot coordinates and geographic itinerary handoff.

The terrain seed is immutable. Floating-origin planning must use the current
frame, even though the session still references its original seed job.
"""
from pyproj import Geod

GEO = Geod(ellps='WGS84')


def active_spec(state, seed_spec):
    return (state.get('origin') or {}).get('spec') or seed_spec


def request_target(target, state, seed_spec):
    """Encode for the advertised private API without moving/restarting a world.

    An already-running legacy API projects against the immutable seed. Until
    its next planned restart, encode the same physical local goal in that seed
    frame. New APIs advertise live_origin and receive geographic targets as-is.
    This compatibility path never changes measured pose or collision checks.
    """
    if state.get('navigation_coordinates') == 'live_origin' or not state.get('origin'):
        return list(target)
    from posim_terrain.terrain import projection
    local = projection(active_spec(state, seed_spec))[1].transform(*target)
    return list(projection(seed_spec)[2].transform(*local))


def target_changed(navigation, target):
    previous = navigation.get('target')
    return not previous or GEO.inv(*previous, *target)[2] > 5.
