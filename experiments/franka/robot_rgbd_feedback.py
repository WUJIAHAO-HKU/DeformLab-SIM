"""Image-only height estimation and a bounded feedback correction.

No simulator state or object nodes are accepted by this controller.
"""
import math
import numpy as np

def estimate_height(rgb,depth,camera_z,focal):
    rgb=np.asarray(rgb);depth=np.asarray(depth)
    if rgb.ndim!=3 or rgb.shape[2]<3 or depth.shape!=rgb.shape[:2]:raise ValueError('RGB-D shape mismatch')
    if not math.isfinite(camera_z) or not math.isfinite(focal) or focal<=0:raise ValueError('Invalid calibration')
    color=rgb[:,:,:3].astype(float)
    mask=(color[:,:,1]>1.5*color[:,:,0])&(color[:,:,2]>1.5*color[:,:,0])&(color[:,:,2]>30)
    # Erode without wraparound so silhouette/background depth cannot enter.
    interior=np.zeros(mask.shape,dtype=bool)
    interior[2:-2,2:-2]=mask[2:-2,2:-2]
    for dy,dx in ((-2,0),(-1,0),(1,0),(2,0),(0,-2),(0,-1),(0,1),(0,2)):
        interior[2:-2,2:-2]&=mask[2+dy:mask.shape[0]-2+dy,2+dx:mask.shape[1]-2+dx]
    valid=interior&np.isfinite(depth)&(depth>.01)&(depth<5.)
    y,x=np.nonzero(valid)
    if len(y)<80:return dict(valid=False,pixels=int(len(y)),height_m=None)
    # This task camera has horizontal optical axis and yaw only in world Z.
    heights=camera_z-(y+.5-depth.shape[0]/2)*depth[y,x]/focal
    return dict(valid=True,pixels=int(len(y)),height_m=float(.5*(np.quantile(heights,.05)+np.quantile(heights,.95))),
                height_iqr_m=float(np.quantile(heights,.75)-np.quantile(heights,.25)))

from robot_height_pi_candidate_v2 import HeightController
