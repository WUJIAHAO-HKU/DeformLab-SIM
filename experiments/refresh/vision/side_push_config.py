"""Prospective geometry-based task settings. No result-dependent constants."""
import numpy as np

CAMERA_POSITION=[.02,-.30,.14]
CAMERA_QUATERNION_WXYZ=[2**-.5,2**-.5,0.,0.]
ROI_MIN=(-.03,-.03,.132);ROI_MAX=(.075,-.005,.148)
SCAN_Z=np.arange(6,dtype=float)*.001+.1375
PUSHER_INITIAL=np.asarray([-.061,0.,.14])

def smooth(x):
    x=float(np.clip(x,0.,1.));return x*x*x*(10+x*(-15+6*x))

def schedule(t,initial_x_offset=0.):
    if not 0<=t<=4.4+1e-8:raise ValueError('Outside task horizon')
    t=min(float(t),3.2)
    u=smooth((t-.4)/1.6);w=smooth((t-2.4)/.4)
    return dict(target_position_m=.02+initial_x_offset+(.04-initial_x_offset)*u,
        nominal_displacement_m=(.041-initial_x_offset)*u-.010*w,
        feedback_weight=1.-w,feedback_active=.4<=t<2.4,
        phase='settle'if t<.4 else 'push'if t<2. else 'hold'if t<2.4 else 'withdraw'if t<2.8 else 'free')

def target_pose(t,correction,initial_x_offset=0.):
    s=schedule(t,initial_x_offset);pos=PUSHER_INITIAL.copy();pos[0]+=initial_x_offset
    pos[0]+=s['nominal_displacement_m']+s['feedback_weight']*float(correction)
    # Purely command-side safety bound, never measured-node feedback.
    if not -.068-1e-9<=pos[0]<=-.015+1e-9:raise ValueError('Command outside predetermined envelope')
    return pos

def settings():
    return dict(task='frictionless finite-support planar side-push alignment',duration_s=3.2,hz=960,
        gravity_m_s2=[0.,0.,-9.81],friction=0.,material_damping=0.,
        phases_s=[0.,.4,2.,2.4,2.8,3.2],primary_window_s=[2.1,2.4],post_withdrawal_window_s=[2.8,3.2],
        target_front_boundary_x_m=.06,auxiliary_success_tolerance_m=.002,release_clearance_threshold_m=.001,
        support_half_extents_xy_m=[.08,.08],nominal_target_margin_m=.020,
        feedback_period_s=.1,gain_per_s=1.2,maximum_correction_m=.004,maximum_slew_m_s=.005,
        scan_z_m=SCAN_Z.tolist(),image_bin_halfwidth_m=.0005,minimum_valid_lines=5,
        image_roi_min_m=list(ROI_MIN),image_roi_max_m=list(ROI_MAX),
        camera_position=CAMERA_POSITION,camera_quaternion_wxyz=CAMERA_QUATERNION_WXYZ,
        initial_calibration='initial nominal right boundary plus raw displacement; initial invalid permanently unavailable, no late fit or truth fallback',
        primary='all-physics-sample mean absolute fixed-front right-boundary error in 2.1--2.4s; invalid feature reported separately, never silently dropped',
        secondary=['post-withdrawal error and drift','residual velocity','actual pusher clearance','support-domain minimum margin','support and side-tool penetration separately','tet J and actual row counts','sensor bias/visibility/invalid rate'],
        scope='Same static-support D binary; moving side-pusher handled by native contact. No tangential-friction benefit or successful settling is assumed.')
