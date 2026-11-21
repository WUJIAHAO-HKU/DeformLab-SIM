"""Prospective finite-support frictional transport; original commands unchanged.

Coefficient choice precedes any side-push physics execution or B/D comparison.
The Coulomb estimate is a feasibility calculation, not native validation.
"""
from side_push_config import *
from side_push_config import settings as original_settings

FRICTION = .4

def settings():
    s = original_settings()
    s.update(task='finite-support frictional planar side-push alignment',
             friction=FRICTION,
             scope='Same static-support D binary; native friction and native side-pusher contact. Real coefficient/combine readback and retained post-withdrawal drift are required.',
             revision='v3 prospective friction revision; original zero-friction protocols preserved and never physically executed')
    return s

def controllability_estimate():
    import math
    # Largest displacement: +41 mm from the -2 mm initial-offset condition.
    length=.043; duration=1.6; feedback_slew=.005; gravity=9.81
    nominal_peak=1.875*length/duration
    speed=nominal_peak+feedback_slew
    peak_acceleration=10/math.sqrt(3)*length/duration**2
    modes={name:(FRICTION**2 if name=='multiply' else FRICTION)
           for name in ('average','minimum','multiply','maximum')}
    return dict(scope='Rigid Coulomb sliding approximation; compliant/contact/controller transients require actual retained tests',
        max_nominal_velocity_m_s=nominal_peak,feedback_slew_m_s=feedback_slew,
        conservative_velocity_m_s=speed,max_nominal_acceleration_m_s2=peak_acceleration,
        modes={name:dict(effective_mu=mu,stop_distance_m=speed**2/(2*mu*gravity),
                        stop_time_s=speed/(mu*gravity),friction_acceleration_m_s2=mu*gravity)
               for name,mu in modes.items()},target_tolerance_m=.002,hold_wait_s=.1,
        caveats=['No guarantee for deformable vibration, sampled-controller discontinuities or actual contact loss',
                 'All same coefficients do not imply the same combined value under multiply',
                 'Actual native coefficients and combine mode must be read independently'])
