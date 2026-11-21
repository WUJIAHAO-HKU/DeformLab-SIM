"""Shared physical XY feedback. Truth is an explicit diagnostic route only."""
import numpy as np
from side_push_config import schedule,PUSHER_INITIAL


def target_pose(t,correction,initial_x_offset=0.):
    c=np.asarray(correction,float)
    if c.shape!=(2,)or not np.isfinite(c).all():raise ValueError('Two finite corrections required')
    s=schedule(t,initial_x_offset);p=PUSHER_INITIAL.copy()
    p[0]+=initial_x_offset+s['nominal_displacement_m']+s['feedback_weight']*c[0]
    # Hold the lateral correction while withdrawing along X: do not drag the
    # object back by returning Y before normal contact has disappeared.
    p[1]+=c[1]
    if not -.068-1e-9<=p[0]<=-.015+1e-9 or abs(p[1])>.00800001:raise ValueError('Actuator envelope')
    return p


class XYObserver:
    def __init__(self,case):
        self.mode=case['xy_control']['observation'];self.initial=np.array([case['initial_x_offset_m'],0.])
        self.nominal=np.array([.04,.04*case['geometry_y_scale']]);self.offset=None;self.exterior_pad=None;self.attempted=False
    def _finish(self,center,half,t,sequence,valid,source):
        center=np.asarray(center,float);half=np.asarray(half,float)
        if not self.attempted:
            if sequence!=0 or abs(t)>1e-12:raise ValueError('Initial calibration clock')
            self.attempted=True
            if valid:
                self.offset=self.initial-center
                self.exterior_pad=np.maximum(self.nominal/2-half,0.)
        valid=bool(valid and self.offset is not None)
        if valid:
            center=center+self.offset
            if self.mode=='rgbd':
                half=half+self.exterior_pad
                # A partial front face does not reveal the rear Y boundary.
                # Use the declared rest extent and a fixed2% extent reserve.
                half[1]=self.nominal[1]*.51
            bounds=np.stack([center-half,center+half])
        else:center=np.zeros(2);bounds=None
        return dict(valid=valid,center_xy_m=center.tolist()if valid else None,
            bounds_xy_m=bounds.tolist()if valid else None,sample_time_s=float(t),sequence=int(sequence),
            observer=source,uses_material_nodes=self.mode=='truth',
            extent_model='Visible right X boundary and known rest X extent, with initial calibration; known rest Y extent with2% reserve'if self.mode=='rgbd'else'exact current node bounds')
    def truth(self,nodes,t,sequence):
        if self.mode!='truth':raise RuntimeError('Truth forbidden in visual route')
        x=np.asarray(nodes,float);valid=x.ndim==2 and x.shape[1]==3 and len(x)>0 and np.isfinite(x).all()
        if not valid:return self._finish([0,0],[0,0],t,sequence,False,'truth_diagnostic')
        lo=x[:,:2].min(0);hi=x[:,:2].max(0)
        return self._finish((lo+hi)/2,(hi-lo)/2,t,sequence,True,'truth_diagnostic')
    def rgbd(self,rgb,depth,calibration,t,sequence):
        if self.mode!='rgbd':raise RuntimeError('Use declared truth diagnostic route')
        d=np.asarray(depth,float);c=np.asarray(rgb)[...,:3].astype(float)
        if d.shape!=(calibration.height,calibration.width)or c.shape!=(*d.shape,3):raise ValueError('Sensor dimensions')
        yy,xx=np.indices(d.shape);valid=np.isfinite(d)&(d>.01)&(d<5.)
        safe=np.where(valid,d,0.)
        camera=np.stack(((xx+.5-calibration.cx)*safe/calibration.fx,(yy+.5-calibration.cy)*safe/calibration.fy,safe),axis=-1)
        m=calibration.matrix();world=camera@m[:3,:3].T+m[:3,3]
        mask=valid&(c[:,:,1]>1.5*c[:,:,0])&(c[:,:,2]>1.5*c[:,:,0])&(c[:,:,2]>30)
        mask &= np.all(world>=[-.05,-.055,.121],axis=-1)&np.all(world<=[.09,.055,.175],axis=-1)
        points=world[mask]
        if len(points)<100:return self._finish([0,0],[0,0],t,sequence,False,'rgbd_foreground')
        lo,hi=np.quantile(points[:,:2],[.002,.998],axis=0)
        center=(lo+hi)/2;center[0]=hi[0]-self.nominal[0]/2;center[1]=np.median(points[:,1]);half=(hi-lo)/2;half[0]=self.nominal[0]/2
        result=self._finish(center,half,t,sequence,True,'rgbd_foreground');result['selected_pixels']=len(points)
        return result
    def state(self):
        return dict(mode=self.mode,fit_count=int(self.offset is not None),updates_after_initial=0,
            offset_xy_m=self.offset.tolist()if self.offset is not None else None,
            initial_exterior_pad_m=self.exterior_pad.tolist()if self.exterior_pad is not None else None)


class XYController:
    def __init__(self,spec):
        self.spec=spec;self.correction=0.;self.correction_y=0.;self.previous=None;self.last_sequence=-1
        self.descent_authorized=False;self.gate_record=None
    def update(self,t,s,observation,gauge_spec):
        if self.previous is not None and abs(t-self.previous-.1)>1e-7:raise ValueError('Feedback clock')
        self.previous=t;seq=observation['sequence'];age=t-observation['sample_time_s']
        valid=bool(observation['valid']and seq>self.last_sequence and -1e-9<=age<=.100001)
        self.last_sequence=max(seq,self.last_sequence)
        old=np.array([self.correction,self.correction_y]);target=np.array([s['target_position_m']-.02,0.])
        request=np.zeros(2);error=None
        if valid:
            error=target-np.asarray(observation['center_xy_m'])
            if s['feedback_active']:request=np.asarray(self.spec['gain_xy_per_s'])*.1*error
        limit=np.asarray(self.spec['action_limit_xy_m']);new=np.clip(old+np.clip(request,-.0005,.0005),-limit,limit)
        self.correction,self.correction_y=map(float,new)
        margin=None
        if valid:
            bounds=np.asarray(observation['bounds_xy_m']);center=np.asarray(gauge_spec['center_xy_m']);half=np.asarray(gauge_spec['opening_xy_m'])/2
            margin=float(np.minimum(bounds[0]-(center-half),(center+half)-bounds[1]).min())
        if abs(t-3.1)<1e-8:
            lower=margin-(self.spec['visual_bound_reserve_m']if observation['observer']=='rgbd_foreground'else 0.)if margin is not None else None
            self.descent_authorized=bool(valid and lower is not None and lower>=.00005)
            self.gate_record=dict(time_s=t,mode=self.spec['gate_mode'],authorized=self.descent_authorized,
                estimated_margin_m=margin,lower_bound_m=lower,requirement_m=.00005,observation_source=observation['observer'])
        return dict(time_s=t,target_position_m=s['target_position_m'],target_center_xy_m=target.tolist(),
            observed_position_m=observation['center_xy_m'][0]if valid else None,observed_center_xy_m=observation['center_xy_m'],
            observation_valid=valid,observation_age_s=age,software_sequence=seq,active=s['feedback_active'],
            error_xy_m=error.tolist()if error is not None else None,correction_m=self.correction,correction_y_m=self.correction_y,
            correction_change_m=self.correction-old[0],correction_y_change_m=self.correction_y-old[1],
            command_effect='next_physics_step',controller='shared_XY_center_feedback',estimated_aperture_margin_m=margin,
            uses_material_nodes=observation['uses_material_nodes'])


def gauge_effective_time(t,controller):
    if controller.spec['gate_mode']=='enabled'and t>3.2 and not controller.descent_authorized:return 3.2
    return t
