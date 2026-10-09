"""Select buffered arrival water from numeric terrain and optional OSM land."""
import hashlib,json,time,math

def candidate(folder,port,radius=350.,coast=None):
 import numpy as np
 from scipy.ndimage import maximum_filter
 from .navigation import TerrainField
 from pyproj import Geod
 values=np.load(folder/'projected.npz');xs,ys,h=values['xs'],values['ys'],values['elevation']
 # Include every vertex of every triangle intersecting the buffered footprint.
 rx=int(np.ceil(radius/min(np.diff(xs))))+1;ry=int(np.ceil(radius/min(np.diff(ys))))+1
 upper=maximum_filter(np.where(np.isfinite(h),h,np.inf),size=(2*ry+1,2*rx+1),mode='constant',cval=np.inf)
 good=np.argwhere(upper< -3.)
 if not len(good):return None
 geo=Geod(ellps='WGS84');lat,lon=values['latitude'],values['longitude']
 field=TerrainField(xs,ys,h)
 meta=json.loads((folder/'manifest.json').read_text())
 shore=None
 if coast is not None:
  import shapely
  from shapely.ops import transform
  from posim_terrain.terrain import projection
  import pyogrio
  bounds=(float(lon.min())-.01,float(lat.min())-.01,float(lon.max())+.01,float(lat.max())+.01)
  metadata,_,geometries,_=pyogrio.raw.read(str(coast.dataset),bbox=bounds,columns=[])
  if metadata['crs']!='EPSG:4326':raise RuntimeError('unexpected_coast_projection')
  land=shapely.union_all(shapely.intersection(shapely.from_wkb(geometries),shapely.box(*bounds)))
  forward=projection(meta['spec'])[1]
  shore=transform(forward.transform,shapely.make_valid(land))
  coverage=transform(forward.transform,shapely.box(*bounds))
  shapely.prepare(shore)
 for j,i in sorted(good,key=lambda ji:geo.inv(port['longitude'],port['latitude'],float(lon[tuple(ji)]),float(lat[tuple(ji)]))[2]):
  floor=field.maximum(float(xs[i]),float(ys[j]),radius)
  if floor is None or floor>=-3.:continue
  if shore is not None:
   point=shapely.Point(float(xs[i]),float(ys[j]));footprint=point.buffer(radius)
   if not coverage.covers(footprint) or shore.intersects(footprint):continue
   shore_distance=float(point.distance(coverage.boundary) if shore.is_empty else shore.distance(point))
   if not math.isfinite(shore_distance) or shore_distance<radius:continue
  break
 else:return None
 meta=json.loads((folder/'manifest.json').read_text());position=[float(lon[j,i]),float(lat[j,i])]
 return dict(longitude=position[0],latitude=position[1],catalog_distance_m=geo.inv(port['longitude'],port['latitude'],*position)[2],
  buffered_floor_max_m=floor,buffer_radius_m=radius,source=meta['source'],source_counts=meta['source_counts'],
  manifest_sha256=hashlib.sha256((folder/'manifest.json').read_bytes()).hexdigest(),
  **(dict(shoreline_distance_m=shore_distance,shoreline_distance_basis='lower_bound_to_read_extent' if shore.is_empty else 'OSM_land_distance') if shore is not None else {}),
  centre_depth_m=-float(h[j,i]),screened_at=time.time(),
  **(dict(arrival_region_radius_m=300.,arrival_region_margin_m=radius-300.) if radius>=300. else {}))
