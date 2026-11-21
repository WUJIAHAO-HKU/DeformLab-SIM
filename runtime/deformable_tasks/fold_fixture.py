"""Deterministic solid support for the separately configured tabletop fold task."""
import numpy as np

def support_geometry(rest,bench_top_m,rest_offset_m=.0008,bar_width_m=.008,bar_clearance_m=.003):
 r=np.asarray(rest,dtype=float)
 if r.ndim!=2 or r.shape[1]!=3 or len(r)<4 or not np.isfinite(r).all():raise ValueError('Finite sheet geometry required')
 if not all(np.isfinite(v) for v in (bench_top_m,rest_offset_m,bar_width_m,bar_clearance_m)) or min(rest_offset_m,bar_width_m,bar_clearance_m)<=0:raise ValueError('Positive support clearances required')
 if np.ptp(r[:,2])>1e-7:raise ValueError('Initially horizontal sheet required')
 low=r.min(0);high=r.max(0);inset=bar_width_m/2+bar_clearance_m
 lower=np.array([low[0]-.01,low[1]+inset,bench_top_m]);upper=np.array([high[0]+.01,high[1]-inset,float(r[0,2])-rest_offset_m])
 if np.any(upper<=lower):raise ValueError('Support cannot fit below sheet and between fixture bars')
 return dict(center_m=((lower+upper)/2).tolist(),size_m=(upper-lower).tolist(),lower_m=lower.tolist(),upper_m=upper.tolist(),cloth_initial_clearance_m=rest_offset_m,bar_clearance_m=bar_clearance_m)

def penetration_depth(points,geometry):
 p=np.asarray(points,float);lo=np.asarray(geometry['lower_m']);hi=np.asarray(geometry['upper_m'])
 if p.ndim!=2 or p.shape[1]!=3 or not np.isfinite(p).all():raise ValueError('Finite positions required')
 inside=np.all((p[:,:2]>=lo[:2])&(p[:,:2]<=hi[:2]),axis=1)
 # A point that has crossed the whole slab must remain a penetration failure.
 return float(np.maximum(hi[2]-p[inside,2],0).max(initial=0))
