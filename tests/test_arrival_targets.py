import json,tempfile,unittest
from pathlib import Path
import numpy as np
from posim_terrain.terrain import projection
from posim_routing.arrival_targets import candidate

class ArrivalTargets(unittest.TestCase):
 def test_entire_arrival_area_keeps_depth_and_land_clearance(self):
  spec=dict(latitude=24.44,longitude=118.08,width_m=4000,height_m=4000,samples=129)
  _,_,inverse=projection(spec);axis=np.linspace(-2000,2000,129)
  xx,yy=np.meshgrid(axis,axis);lon,lat=inverse.transform(xx,yy);h=np.where(xx>=0,5.,-8.)
  with tempfile.TemporaryDirectory() as folder:
   p=Path(folder);np.savez(p/'projected.npz',xs=axis,ys=axis,elevation=h,longitude=lon,latitude=lat)
   (p/'manifest.json').write_text(json.dumps(dict(spec=spec,source='fixture',source_counts={'fixture':129**2})))
   value=candidate(p,dict(longitude=spec['longitude'],latitude=spec['latitude']),350.)
   self.assertGreater(value['catalog_distance_m'],350.)
   self.assertEqual(value['arrival_region_radius_m'],300.)
   self.assertEqual(value['arrival_region_margin_m'],50.)
   self.assertLess(value['buffered_floor_max_m'],-3.)

 def test_unknown_coverage_cannot_supply_arrival_water(self):
  with tempfile.TemporaryDirectory() as folder:
   p=Path(folder);axis=np.linspace(-500,500,65);h=np.full((65,65),np.nan)
   np.savez(p/'projected.npz',xs=axis,ys=axis,elevation=h,longitude=np.zeros(h.shape),latitude=np.zeros(h.shape))
   self.assertIsNone(candidate(p,dict(longitude=0.,latitude=0.)))

 def test_open_water_without_land_in_read_extent_has_an_explicit_finite_lower_bound(self):
  from unittest.mock import patch
  from types import SimpleNamespace
  import math
  spec=dict(latitude=35.,longitude=129.,width_m=1000,height_m=1000,samples=65)
  _,_,inverse=projection(spec);axis=np.linspace(-500,500,65);xx,yy=np.meshgrid(axis,axis);lon,lat=inverse.transform(xx,yy)
  with tempfile.TemporaryDirectory() as folder:
   p=Path(folder);h=np.full(xx.shape,-8.);np.savez(p/'projected.npz',xs=axis,ys=axis,elevation=h,longitude=lon,latitude=lat)
   (p/'manifest.json').write_text(json.dumps(dict(spec=spec,source='fixture',source_counts={'fixture':65**2})))
   with patch('pyogrio.raw.read',return_value=({'crs':'EPSG:4326'},None,np.array([],dtype=object),None)):
    value=candidate(p,dict(latitude=35.,longitude=129.),350.,SimpleNamespace(dataset='fixture'))
   self.assertTrue(math.isfinite(value['shoreline_distance_m']))
   self.assertGreater(value['shoreline_distance_m'],350.)
   self.assertEqual(value['shoreline_distance_basis'],'lower_bound_to_read_extent')

if __name__=='__main__':unittest.main()
