"""Conservative cached terrain upper bounds for the existing exact A* planner."""
import math
import numpy as np
from scipy.ndimage import maximum_filter

class CachedTerrain:
 def __init__(self,field):
  self.field=field;self.filters={}
  self.values=np.where(np.isfinite(field.heights),field.heights,math.inf)
 def maximum(self,x,y,radius):
  xs,ys=self.field.xs,self.field.ys
  if x-radius<xs[0] or x+radius>xs[-1] or y-radius<ys[0] or y+radius>ys[-1]:return None
  i0=max(0,int(np.searchsorted(xs,x-radius,side='right'))-1)
  i1=int(np.searchsorted(xs,x+radius,side='left'))
  j0=max(0,int(np.searchsorted(ys,y-radius,side='right'))-1)
  j1=int(np.searchsorted(ys,y+radius,side='left'))
  nx,ny=i1-i0+1,j1-j0+1
  key=(ny,nx)
  if key not in self.filters:self.filters[key]=maximum_filter(self.values,size=key,mode='constant',cval=math.inf)
  bound=float(self.filters[key][j0+ny//2,i0+nx//2])
  if not math.isfinite(bound):return None
  # A planar triangle never exceeds its highest vertex. An entirely deep
  # vertex rectangle proves the swept footprint safe without clipping every
  # triangle. Near land, retain the original exact clipping calculation.
  return bound if bound< -1.5 else self.field.maximum(x,y,radius)
