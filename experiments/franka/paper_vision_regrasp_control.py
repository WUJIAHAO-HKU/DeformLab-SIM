"""Image-only release/reacquire/regrasp state machine; object truth is excluded."""
import math
import numpy as np
from paper_contact_vision_control import ContactController,smooth

class RegraspController:
    def __init__(self,mode):
        if mode!='rgbd':raise ValueError('Regrasp requires an image-only RGB-D policy')
        self.transport=ContactController(mode)
        self.visual_anchor=None;self.anchor_time=None;self.reacquired=[];self.goal=None
        self.last_time=None;self.invalid_reacquisition_frames=0;self.out_of_range_frames=0
    @property
    def correction(self):return self.transport.correction
    @property
    def updates(self):return self.transport.updates
    @property
    def anchor(self):return self.transport.anchor
    def observe(self,t,estimate):
        self.transport.observe(t,estimate);self.last_time=t
        valid=estimate.get('valid') and estimate.get('x_m') is not None and math.isfinite(estimate['x_m'])
        if t<=.5 and valid:
            self.visual_anchor=float(estimate['x_m']);self.anchor_time=t
        if 7.5<=t<=9.5:
            if valid and self.visual_anchor is not None:
                delta=float(estimate['x_m'])-self.visual_anchor
                if abs(delta)<=.060:
                    self.reacquired.append(delta)
                    if len(self.reacquired)>=3:self.goal=float(np.median(self.reacquired[-5:]))
                else:self.out_of_range_frames+=1
            else:self.invalid_reacquisition_frames+=1
        return self.correction
    def state(self):
        return dict(initial_visual_anchor_x_m=self.visual_anchor,initial_visual_anchor_time_s=self.anchor_time,
            valid_reacquisition_samples=len(self.reacquired),regrasp_ready=self.goal is not None,
            regrasp_target_x_offset_m=self.goal,transport_correction_m=self.correction,
            invalid_reacquisition_frames=self.invalid_reacquisition_frames,out_of_range_frames=self.out_of_range_frames)
    def motion(self,t,actuation_offset,side):
        # The bias belongs to the initial transport plant, never to the observer.
        transport_x=(.020+actuation_offset)*smooth((t-2.5)/2.)+self.correction
        if t<=.5:opening=.04
        elif t<=2.:opening=.04-(.04-.3*side)*(t-.5)/1.5
        elif t<=5.5:opening=.3*side
        elif t<=6.5:opening=.3*side+(.04-.3*side)*(t-5.5)
        else:opening=.04
        rise=-.001*smooth((t-2.)/.5)
        lateral=transport_x
        if t>6.5:rise=-.001+.051*smooth((t-6.5)/1.)
        if t>7.5 and self.goal is not None:
            lateral=transport_x+(self.goal-transport_x)*smooth((t-7.5)/1.)
        if t>9.5:rise=.05-.051*smooth((t-9.5)/1.)
        if t>10.5 and self.goal is not None:opening=.04-(.04-.3*side)*smooth((t-10.5)/1.)
        if t>11.5:rise=-.001+.041*smooth((t-11.5)/1.)
        return [float(opening),float(rise),float(lateral)]

def score(time,nodes,hand_positions,table_top):
    t=np.asarray(time,float);t=t-t[0];x=np.asarray(nodes,float);hand=np.asarray(hand_positions,float)
    if len(t)!=12961 or x.shape[0]!=len(t) or hand.shape!=(len(t),3) or not np.isfinite(x).all():raise ValueError('Incomplete regrasp trajectory')
    if not np.allclose(np.diff(t),1/960.,atol=1e-9):raise ValueError('Unexpected clock')
    centroid=x.mean(axis=1);reference=centroid[7680,2]
    window=(t>=12.5-1e-9)&(t<=13.5+1e-9)
    lift=centroid[window,2]-reference;clearance=x[window,:,2].min(axis=1)-table_top
    relative=centroid[window]-hand[window];drift=np.linalg.norm(relative-relative[0],axis=1)
    return dict(reference_time_s=8.,hold_window_s=[12.5,13.5],minimum_centroid_lift_m=float(lift.min()),
        minimum_all_node_support_clearance_m=float(clearance.min()),maximum_object_hand_relative_drift_m=float(drift.max()),
        unsupported_lift_success=bool(lift.min()>=.025 and clearance.min()>=.010 and drift.max()<=.005),
        centroid_definition='Arithmetic mean of native nodal positions; offline only',
        scope='Physical unsupported hold after release/reacquisition; separate collider-release audit is required')
