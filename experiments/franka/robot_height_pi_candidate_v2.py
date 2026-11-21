"""Development-only bounded PI height feedback; no simulator-state access.

The observer supplies a valid height and a timestamp. Gains are common to both
backends. This module is not registered in a frozen experiment or live platform.
"""
import math


def clip(value, low, high):
    return min(high, max(low, value))


class HeightController:
    def __init__(self, mode, goal_height=None):
        if mode not in ('open', 'rgbd'):
            raise ValueError('Unknown feedback mode')
        if goal_height is None or not math.isfinite(goal_height):
            raise ValueError('Finite observation-space goal required')
        self.mode=mode
        self.goal_height=float(goal_height)
        self.anchor=None
        self.correction=0.
        self.integral=0.
        self.updates=0
        self.last_time=None
        self.kp=2.
        self.ki=6.
        self.max_rate=.02
        self.max_correction=.025

    def observe(self, time_s, estimate, lift_height):
        if not math.isfinite(time_s) or (self.last_time is not None and time_s<=self.last_time):
            raise ValueError('Strictly increasing finite timestamps required')
        dt=0. if self.last_time is None else time_s-self.last_time
        self.last_time=time_s
        applied=False
        error=target=None
        limited=False
        valid=estimate.get('valid') is True
        if valid and not math.isfinite(estimate['height_m']):
            raise ValueError('Valid observation must be finite')
        if valid:
            if 2.49<=time_s<=2.51:
                self.anchor=float(estimate['height_m'])
            if self.anchor is not None and 2.49<=time_s<=5.5:
                u=clip((time_s-2.5)/2.,0.,1.)
                fraction=u*u*u*(10.+u*(-15.+6.*u))
                target=self.anchor+(self.goal_height-self.anchor)*fraction
                error=target-estimate['height_m']
                if self.mode=='rgbd':
                    # A stale sample must not accumulate unseen error over a long gap.
                    if dt>.15:
                        return self.record(target,error,False,False)
                    proposed_integral=self.integral+self.ki*dt*error
                    raw=self.kp*error+proposed_integral
                    delta=clip(raw-self.correction,-self.max_rate*dt,self.max_rate*dt)
                    bounded=clip(self.correction+delta,-self.max_correction,self.max_correction)
                    limited=abs(raw-bounded)>1e-12
                    # Track the command actually applied: no hidden integral windup.
                    self.integral=bounded-self.kp*error
                    self.correction=bounded
                    self.updates+=1
                    applied=True
        return self.record(target,error,applied,limited)

    def record(self,target,error,applied,limited):
        return dict(anchor_m=self.anchor,target_height_m=target,error_m=error,
                    correction_m=self.correction,update_applied=applied,
                    integral_m=self.integral,command_limited=limited,
                    controller_revision='bounded-pi-development-v2')
