"""Error-controlled dyadic refinement of the endpoint impact reference.

Step doubling compares complete position/velocity states. Only accepted fine
steps contribute to force, impulse, work, or time. No Richardson extrapolation,
velocity filtering, or hidden penetration projection is used.
"""
import numpy as np
from .coupled_coupon_reference import ImpactCoupledCoupon
from .midpoint_coupon_reference import HybridMidpointCoupon
from .hybrid_gauss_coupon import HybridGaussCoupon

class AdaptiveCoupledCoupon:
    def __init__(self,*args,target_function,target_derivative=None,position_tolerance=1e-8,velocity_tolerance=2e-5,max_level=16,integrator='backward-euler',**kwargs):
        if position_tolerance<=0 or velocity_tolerance<=0:raise ValueError('Positive adaptive tolerances required')
        if integrator not in ('backward-euler','midpoint-hybrid','gauss-hybrid'):raise ValueError('Unknown reference integrator')
        if integrator=='gauss-hybrid' and target_derivative is None:raise ValueError('Gauss hybrid requires explicit target derivative')
        self.target_derivative=target_derivative
        self.integrator=integrator
        self.args=list(args);self.kwargs=kwargs;self.hz=float(args[5]);self.dt=1/self.hz
        self.target_function=target_function;self.position_tolerance=position_tolerance;self.velocity_tolerance=velocity_tolerance
        self.max_level=max_level;self.models={};self.base=self.model(0)
        for name in ('rest','tets','n','mass','vol','M','B','g0','top','bottom'):setattr(self,name,getattr(self.base,name))
        self.u=self.base.u.copy();self.v=self.base.v.copy();self.target=0.;self.step_index=0;self.level=0
        self.last_substeps=[];self.total_accepted=0;self.total_rejected=0

    def model(self,level):
        if level not in self.models:
            args=self.args.copy();args[5]=self.hz*2**level
            cls={'backward-euler':ImpactCoupledCoupon,'midpoint-hybrid':HybridMidpointCoupon,'gauss-hybrid':HybridGaussCoupon}[self.integrator]
            self.models[level]=cls(*args,**self.kwargs)
        return self.models[level]

    def advance(self,level,t,u,v):
        model=self.model(level);model.u=u.copy();model.v=v.copy();model.target=self.target_function(t)
        try:
            if self.integrator=='gauss-hybrid':
                row=model.step(self.target_function(t+model.dt),self.target_derivative(t),self.target_derivative(t+model.dt))
            else:row=model.step(self.target_function(t+model.dt))
        except Exception:
            self.failure_state=dict(time_s=t,trial_dt_s=model.dt,u_start=u,v_start=v,target_end=self.target_function(t+model.dt))
            raise
        row['time_s']=t+model.dt;row['dt_s']=model.dt
        return model.u.copy(),model.v.copy(),row

    def step(self,target):
        start=self.step_index*self.dt
        if abs(target-self.target_function(start+self.dt))>1e-12:raise ValueError('Target function and frame target differ')
        total_ticks=2**self.max_level;cursor=0;accepted=[];rejections=0;errors=[]
        u=self.u.copy();v=self.v.copy();level=self.level
        while cursor<total_ticks:
            while 2**(self.max_level-level)>total_ticks-cursor:level+=1
            stride=2**(self.max_level-level);t=start+cursor*self.dt/total_ticks
            uc,vc,_=self.advance(level,t,u,v)
            uh,vh,r1=self.advance(level+1,t,u,v)
            uf,vf,r2=self.advance(level+1,t+self.dt/2**(level+1),uh,vh)
            error=max(float(abs(uf-uc).max()/self.position_tolerance),float(abs(vf-vc).max()/self.velocity_tolerance))
            if not np.isfinite(error):raise RuntimeError('Nonfinite adaptive error')
            if error>1:
                rejections+=1
                if level>=self.max_level:
                    self.failure_state=dict(time_s=t,trial_dt_s=self.dt/2**level,error=error,u_start=u,v_start=v,
                        u_coarse=uc,v_coarse=vc,u_fine=uf,v_fine=vf,gap_coarse=self.g0+self.B@uc,gap_fine=self.g0+self.B@uf)
                    raise RuntimeError(f'Adaptive error tolerance unmet at minimum step: t={t:.12g}, h={self.dt/2**level:.6g}, error={error:.6g}')
                level+=1;continue
            accepted.extend((r1,r2));errors.append(error);u=uf;v=vf;cursor+=stride
            if error<.2 and level>0:level-=1
        self.u=u;self.v=v;self.target=target;self.step_index+=1;self.level=level
        self.total_accepted+=len(accepted);self.total_rejected+=rejections;self.last_substeps=accepted
        row=accepted[-1].copy();row.pop('dt_s');row['step']=self.step_index;row['time_s']=self.step_index*self.dt
        sums=('kinetic_change_J','elastic_change_J','drive_work_J','contact_work_J','material_drag_dissipation_J',
            'integration_loss_J','mechanical_work_residual_J','drive_target_work_J','drive_spring_change_J',
            'drive_dissipation_J','total_integration_loss_J','full_work_residual_J','impact_work_J',
            'impact_kinetic_jump_J','impact_work_residual_J','impact_top_impulse_Ns','impact_bottom_impulse_Ns',
            'soft_drag_impulse_z_Ns','contact_angular_impulse_x_Nms','contact_angular_impulse_y_Nms','contact_angular_impulse_z_Nms')
        for key in sums:row[key]=sum(r[key] for r in accepted)
        for key in ('drive_force_N','reaction_N','bottom_reaction_N'):row[key]=sum(r[key]*r['dt_s'] for r in accepted)/self.dt
        for key in ('max_penetration_m','force_residual_N','natural_residual_m','iterations',
            'impact_impulse_residual_Ns','impact_iterations','closed_inward_speed_m_s','position_integrated_velocity_residual_m',
            'linearized_angular_impulse_residual_Nms','peak_preimpact_nodal_speed_m_s','internal_force_sum_N'):
            row[key]=max(abs(r[key]) for r in accepted)
        row['min_J']=min(r['min_J'] for r in accepted)
        if self.integrator in ('midpoint-hybrid','gauss-hybrid'):
            row['midpoint_step']=sum(r['midpoint_step'] for r in accepted)
            row['contact_activation_fallback']=sum(r['contact_activation_fallback'] for r in accepted)
        if self.integrator=='gauss-hybrid':
            for key in ('gauss_step','contact_phase_fallback'):row[key]=sum(r[key] for r in accepted)
            for key in ('gauss_mapping_position_error_m','gauss_mapping_velocity_error_m_s','gauss_impulse_residual_Ns'):
                row[key]=max(r[key] for r in accepted)
        row.update(accepted_substeps=len(accepted),rejected_trials=rejections,max_accepted_error=max(errors),
            minimum_substep_s=min(r['dt_s'] for r in accepted),maximum_substep_s=max(r['dt_s'] for r in accepted),
            peak_substep_nodal_speed_m_s=max(r['max_nodal_speed_m_s'] for r in accepted))
        return row
