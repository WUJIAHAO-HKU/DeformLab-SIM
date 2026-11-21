"""Image-only leading-front boundary observers; depth mode never reads RGB.

Calibration/_points/component implementation is imported from the pinned v5
camera utilities. These functions accept no nodes, truth masks or backend IDs.
"""
import numpy as np
from observers import _points,_largest_component,NominalROI
from side_push_config import ROI_MIN,ROI_MAX,SCAN_Z

ROI=NominalROI(ROI_MIN,ROI_MAX)

def _measure(world,mask,mode,t,seq):
    ij,components=_largest_component(mask);p=world[ij[:,0],ij[:,1]];lines=[]
    for z in SCAN_Z:
        take=abs(p[:,2]-z)<=.0005
        valid=int(take.sum())>=12
        pixel=None;point=None
        if valid:
            k=np.flatnonzero(take)[np.argmax(p[take,0])];pixel=ij[k].tolist();point=p[k].tolist()
        lines.append(dict(z_m=float(z),valid=valid,pixels=int(take.sum()),point_world=point,pixel_yx=pixel))
    values=[line['point_world'][0]for line in lines if line['valid']]
    valid=len(p)>=80 and len(values)>=5
    return dict(observer=mode,valid=valid,position_m=float(np.median(values))if valid else None,
        reason='ok'if valid else 'insufficient_front_patch_at_fixed_heights',
        lines=lines,selected_pixels=len(p),components=components,sample_time_s=float(t),sequence=int(seq),
        hardware_exposure_timestamp_verified=False,measurement='median rightmost observed front-patch X in fixed world-z bands')

def observe_color(rgb,depth,calibration,roi=None,*,sample_time_s,sequence):
    world,mask=_points(depth,calibration,ROI);c=np.asarray(rgb)
    if c.shape[:2]!=mask.shape or c.ndim!=3 or c.shape[2]<3:raise ValueError('RGB shape')
    c=c[:,:,:3].astype(float)
    mask &= (c[:,:,1]>1.5*c[:,:,0])&(c[:,:,2]>1.5*c[:,:,0])&(c[:,:,2]>30)
    return _measure(world,mask,'color_rgbd',sample_time_s,sequence)

def observe_depth(depth,calibration,roi=None,*,sample_time_s,sequence):
    world,roi_mask=_points(depth,calibration,ROI);d=np.asarray(depth,float)
    finite=np.isfinite(d)&(d>.01)&(d<5.)
    dx=world[1:-1,2:]-world[1:-1,:-2];dy=world[2:,1:-1]-world[:-2,1:-1]
    n=np.cross(dx,dy);ln=np.linalg.norm(n,axis=-1)
    view=calibration.matrix()[:3,3]-world[1:-1,1:-1];lv=np.linalg.norm(view,axis=-1)
    cosine=np.divide(abs(np.sum(n*view,axis=-1)),ln*lv,out=np.zeros_like(ln),where=(ln>1e-12)&(lv>0))
    continuous=finite[1:-1,1:-1].copy()
    for sy,sx in ((slice(1,-1),slice(2,None)),(slice(1,-1),slice(None,-2)),(slice(2,None),slice(1,-1)),(slice(None,-2),slice(1,-1))):
        continuous &= finite[sy,sx]&(np.linalg.norm(world[sy,sx]-world[1:-1,1:-1],axis=-1)<=.002)
    mask=np.zeros_like(roi_mask);mask[1:-1,1:-1]=continuous&(cosine>=np.cos(np.pi/6))&roi_mask[1:-1,1:-1]
    interior=np.zeros_like(mask);interior[1:-1,1:-1]=True
    for y in (-1,0,1):
        for x in (-1,0,1):interior[1:-1,1:-1]&=mask[1+y:mask.shape[0]-1+y,1+x:mask.shape[1]-1+x]
    result=_measure(world,interior,'depth_geometry',sample_time_s,sequence)
    result['normal_selection']=dict(max_angle_degrees=30,neighbor_distance_m=.002,erosion_pixels=1)
    return result
