"""Image-only tabletop transport control; no simulator-state arguments."""
import math
import numpy as np

def smooth(u):
    u=float(np.clip(u,0.,1.));return u*u*u*(10.+u*(-15.+6.*u))

def estimate_x(rgb,depth,camera_position,focal,yaw):
    rgb=np.asarray(rgb);depth=np.asarray(depth);camera=np.asarray(camera_position,float)
    if rgb.ndim!=3 or rgb.shape[2]<3 or depth.shape!=rgb.shape[:2]:raise ValueError('RGB-D shape mismatch')
    if camera.shape!=(3,) or not np.isfinite(camera).all() or not math.isfinite(focal) or focal<=0 or not math.isfinite(yaw):raise ValueError('Invalid calibration')
    c=rgb[:,:,:3].astype(float);mask=(c[:,:,1]>1.5*c[:,:,0])&(c[:,:,2]>1.5*c[:,:,0])&(c[:,:,2]>30)
    eroded=np.zeros_like(mask);eroded[2:-2,2:-2]=mask[2:-2,2:-2]
    for dy,dx in ((-2,0),(-1,0),(1,0),(2,0),(0,-2),(0,-1),(0,1),(0,2)):
        eroded[2:-2,2:-2]&=mask[2+dy:mask.shape[0]-2+dy,2+dx:mask.shape[1]-2+dx]
    valid=eroded&np.isfinite(depth)&(depth>.01)&(depth<5.)
    y,x=np.nonzero(valid)
    if len(y)<80:return dict(valid=False,pixels=int(len(y)),x_m=None)
    d=depth[y,x];cam_x=(x+.5-depth.shape[1]/2)*d/focal
    world_x=camera[0]+math.cos(yaw)*cam_x-math.sin(yaw)*d
    return dict(valid=True,pixels=int(len(y)),x_m=float(.5*(np.quantile(world_x,.05)+np.quantile(world_x,.95))))

class ContactController:
    def __init__(self,mode,travel=.020):
        if mode not in ('open','rgbd'):raise ValueError('No privileged feedback in this task')
        self.mode=mode;self.travel=float(travel);self.anchor=None;self.correction=0.;self.integral=0.;self.last_time=None;self.updates=0
    def observe(self,t,estimate):
        if not math.isfinite(t) or (self.last_time is not None and t<=self.last_time):raise ValueError('Nonmonotonic observation clock')
        dt=0. if self.last_time is None else t-self.last_time;self.last_time=t
        if not estimate.get('valid') or not math.isfinite(estimate.get('x_m',float('nan'))):return self.correction
        if t<=2.5:self.anchor=float(estimate['x_m']);return self.correction
        if self.mode=='open' or self.anchor is None or t>5.5:return self.correction
        e=self.anchor+self.travel*smooth((t-2.5)/2.)-estimate['x_m']
        candidate=self.integral+(e*dt if dt<=.15 else 0.)
        raw=2.*e+6.*candidate
        bounded=float(np.clip(raw,-.012,.012));delta=.020*min(dt,.15)
        applied=float(np.clip(bounded,self.correction-delta,self.correction+delta))
        self.integral=(applied-2.*e)/6.;self.correction=applied;self.updates+=1
        return applied

def score(time,nodes,table_top):
    time=np.asarray(time,float);time=time-time[0];nodes=np.asarray(nodes,float)
    if len(time)!=7201 or nodes.shape[0]!=len(time) or not np.isfinite(nodes).all():raise ValueError('Incomplete trajectory')
    center=nodes.mean(axis=1);anchor=int(np.argmin(abs(time-2.5)));target=center[anchor,0]+.020
    window=(time>=5.)&(time<=5.5);contact=(time>=2.5)&(time<=5.5)
    bottom=nodes[:,:,2].min(axis=1)
    err=center[window,0]-target
    return dict(target_x_m=float(target),hold_x_max_error_m=float(abs(err).max()),hold_x_mae_m=float(abs(err).mean()),
        final_x_error_m=float(center[-1,0]-target),support_fraction=float(np.mean(bottom[contact]-table_top<=.001)),
        maximum_tabletop_penetration_m=float(max(0.,table_top-bottom.min())),minimum_table_clearance_m=float((bottom[contact]-table_top).min()),
        visual_target_success=bool(abs(err).max()<=.003),support_success=bool(np.mean(bottom[contact]-table_top<=.001)>=.95),
        target_definition='20 mm world-X displacement from independently measured pre-transport center at 2.5 s; RGB-D controller uses its own image anchor')
