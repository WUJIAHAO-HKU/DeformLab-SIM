"""Second image-only observer: median depth points in the largest visible component.

Inputs are RGB, depth and camera calibration only. No simulator state or control
command enters segmentation, component selection or position estimation.
"""
import math
import numpy as np

def estimate_x(rgb,depth,camera_position,focal,yaw):
    rgb=np.asarray(rgb);depth=np.asarray(depth);camera=np.asarray(camera_position,float)
    if rgb.ndim!=3 or rgb.shape[2]<3 or depth.shape!=rgb.shape[:2]:raise ValueError('RGB-D shape mismatch')
    if camera.shape!=(3,) or not np.isfinite(camera).all() or not math.isfinite(focal) or focal<=0 or not math.isfinite(yaw):raise ValueError('Invalid calibration')
    c=rgb[:,:,:3].astype(float)
    mask=(c[:,:,1]>1.5*c[:,:,0])&(c[:,:,2]>1.5*c[:,:,0])&(c[:,:,2]>30)
    eroded=np.zeros_like(mask);eroded[2:-2,2:-2]=mask[2:-2,2:-2]
    for dy,dx in ((-2,0),(-1,0),(1,0),(2,0),(0,-2),(0,-1),(0,1),(0,2)):
        eroded[2:-2,2:-2]&=mask[2+dy:mask.shape[0]-2+dy,2+dx:mask.shape[1]-2+dx]
    valid=eroded&np.isfinite(depth)&(depth>.01)&(depth<5.)
    yy,xx=np.nonzero(valid);available=set(zip(yy.tolist(),xx.tolist()));components=[]
    # Deterministic tie break: largest component, then first raster-order pixel.
    for pixel in zip(yy.tolist(),xx.tolist()):
        if pixel not in available:continue
        available.remove(pixel);members=[pixel];pending=[pixel]
        while pending:
            y,x=pending.pop()
            for q in ((y-1,x),(y,x-1),(y,x+1),(y+1,x)):
                if q in available:available.remove(q);members.append(q);pending.append(q)
        components.append(members)
    selected=max(components,key=len) if components else []
    info=dict(valid=len(selected)>=80,pixels=len(selected),total_valid_pixels=len(yy),components=len(components),x_m=None)
    if not info['valid']:return info
    ij=np.asarray(selected,int);y,x=ij[:,0],ij[:,1];d=depth[y,x]
    world_x=camera[0]+math.cos(yaw)*(x+.5-depth.shape[1]/2)*d/focal-math.sin(yaw)*d
    info['x_m']=float(np.median(world_x))
    return info
