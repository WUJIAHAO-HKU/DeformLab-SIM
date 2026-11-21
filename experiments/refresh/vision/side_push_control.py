"""Common scalar visual/truth controller, without backend or node inputs."""
import numpy as np

class InitialPositionCalibration:
    def __init__(self,nominal):
        self.nominal=float(nominal);self.attempted=False;self.offset=None
    def apply(self,raw):
        if not self.attempted:
            if raw['sequence']!=0:raise ValueError('Initial capture required')
            self.attempted=True
            if raw['valid'] and raw.get('position_m')is not None and np.isfinite(raw['position_m']):
                self.offset=self.nominal-float(raw['position_m'])
        result=dict(raw)
        if self.offset is None:result.update(valid=False,position_m=None,reason='initial_calibration_unavailable')
        elif raw['valid']:
            if raw.get('position_m')is None or not np.isfinite(raw['position_m']):raise ValueError('Invalid valid measurement')
            result['position_m']=float(raw['position_m'])+self.offset
        return result
    def state(self):
        return dict(attempted=self.attempted,available=self.offset is not None,offset_m=self.offset,
            fit_count=int(self.offset is not None),nominal_position_m=self.nominal,updates_after_initial=0)

class PositionController:
    def __init__(self):self.correction=0.;self.previous=None;self.last_sequence=-1
    def update(self,t,target,observation,*,active=True):
        t=float(t);target=float(target)
        if not np.isfinite([t,target]).all() or t<0:raise ValueError('Invalid request')
        if self.previous is not None and abs(t-self.previous-.1)>1e-7:raise ValueError('Feedback clock')
        self.previous=t;seq=int(observation['sequence']);age=t-float(observation['sample_time_s'])
        x=observation.get('position_m');valid=bool(observation['valid']) and x is not None and np.isfinite(x)
        valid=bool(valid and -1e-9<=age<=.100001 and seq>self.last_sequence);self.last_sequence=max(seq,self.last_sequence)
        old=self.correction;error=target-float(x) if valid else None
        requested=0.
        if valid and active:
            requested=1.2*error*.1
            self.correction=float(np.clip(old+np.clip(requested,-.0005,.0005),-.004,.004))
        return dict(time_s=t,target_position_m=target,observed_position_m=x,observation_valid=valid,
            observation_age_s=age,software_sequence=seq,active=bool(active),error_m=error,
            correction_m=self.correction,correction_change_m=self.correction-old,
            slew_limited=bool(abs(requested)>.0005),offset_limited=bool(abs(self.correction)>=.004),
            command_effect='next_physics_step',hardware_latency_verified=False)
