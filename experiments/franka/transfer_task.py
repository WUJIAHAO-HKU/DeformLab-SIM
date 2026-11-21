"""Common Franka controller and independent mesh scoring for the slot task."""
import numpy as np

HZ=960
END=10.
TRANSLATION=.052
LIFT=.070
RAIL_HEIGHT=.008
INNER_Y=.062

def smooth(u):
    u=np.clip(u,0.,1.)
    return u**3*(10+u*(-15+6*u))

def command(t, side, closed_fraction):
    # Fixed visual-independent reference, tracked through native joint targets.
    # Initial object pose and mesh never enter the controller.
    close=smooth((t-.55)/1.55)-smooth((t-6.65)/.75)
    opening=.04-(.04-closed_fraction*side)*close
    dz=LIFT*(smooth((t-2.2)/1.5)-smooth((t-5.05)/1.5))+.064*smooth((t-7.4)/1.2)
    dx=TRANSLATION*smooth((t-3.75)/1.25)-.025*smooth((t-8.6)/.85)
    return opening,np.array([dx,0.,dz])

def fixture(center,top,side,opening_extra):
    xy=np.array(center[:2])+[TRANSLATION,0.]
    inner=np.array([side/2+opening_extra/2,INNER_Y])
    thickness=.005
    pieces=[]
    for axis in (0,1):
        for sign in (-1,1):
            pos=np.r_[xy,top+RAIL_HEIGHT/2]
            pos[axis]+=sign*(inner[axis]+thickness/2)
            full=np.r_[2*inner,RAIL_HEIGHT];full[axis]=thickness
            if axis==1:full[0]+=2*thickness
            pieces.append((pos,full))
    return xy,inner,pieces

def score(t,x,hand,actual_joints,quality,center,top,xy,inner,rails):
    t=t-t[0]
    assert len(t)==int(END*HZ)+1 and np.allclose(np.diff(t),1/HZ,atol=1e-8)
    centroid=x.mean(1)
    margin=(inner-np.abs(x[:,:,:2]-xy)).min(axis=(1,2))
    low=x[:,:,2].min(axis=1)
    overlap=np.maximum.reduce([np.maximum(0.,(size/2-np.abs(x-pos)).min(axis=2)).max(axis=1) for pos,size in rails])
    lift=(t>=3.7)&(t<=5.0)
    final=(t>=9.5)
    col=(t>=5.0)&(t<=9.5)
    relative=centroid-hand[:,:3]
    lift_drift=np.linalg.norm(relative[lift]-relative[lift][0],axis=1).max()
    moved=float(centroid[np.argmin(abs(t-5.0)),0]-centroid[0,0])
    err=np.linalg.norm(centroid[:,:2]-xy,axis=1)
    speed=np.linalg.norm(np.diff(centroid[final],axis=0),axis=1)*HZ
    criteria=dict(unsupported_pick=bool((low[lift]-top).min()>.035),
        retained_in_gripper=bool(lift_drift<.007),carried_to_slot=bool(moved>.040),
        whole_object_in_opening=bool(margin[final].min()>=0),
        deposited_on_floor=bool(np.max(np.abs(low[final]-top))<=.001),
        released_and_retreated=bool(actual_joints[final,7:].min()>=.038 and
          np.min(hand[final,2]-x[final,:,2].max(axis=1))>=.100),
        stable_after_withdrawal=bool(np.quantile(speed,.95)<.025),
        valid_elements=bool(quality.min()>0),rail_penetration_below_1mm=bool(overlap[col].max()<=.001))
    return dict(success=all(criteria.values()),criteria=criteria,final_center_error_m=float(np.median(err[final])),
       final_minimum_margin_m=float(margin[final].min()),peak_sampled_rail_penetration_m=float(overlap[col].max()),
       minimum_lift_support_clearance_m=float((low[lift]-top).min()),lift_relative_drift_m=float(lift_drift),
       carry_m=moved,minimum_J=float(quality.min()),settled_speed_p95_m_s=float(np.quantile(speed,.95)),
       final_floor_height_error_m=float(np.max(abs(low[final]-top))),
       minimum_hand_to_block_top_m=float(np.min(hand[final,2]-x[final,:,2].max(axis=1)))),dict(margin_m=margin,rail_penetration_m=overlap,center_error_m=err)
