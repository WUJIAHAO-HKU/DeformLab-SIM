"""Fourth-order smooth contact phases with audited midpoint/impact transitions.

Independent linear-FEM reference only. Changing active sets use the existing
position/impact solver; step doubling controls the complete mixed update.
"""
from collections import OrderedDict
import numpy as np
from .midpoint_coupon_reference import HybridMidpointCoupon
from .frozen_contact_oracle import FrozenContactOracle
from .driven_gauss_reference import DrivenGaussStepper,ContactPhaseExit


class HybridGaussCoupon(HybridMidpointCoupon):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs);self.gauss_cache=OrderedDict()

    def step(self,target,rate_start=0.,rate_end=0.):
        gap=self.g0+self.B@self.u
        active=gap<=max(5e-17,2*max(5e-18,min(1e-11,self.dt*1e-9)))
        try:
            if np.max(abs((self.B@self.v)[active]),initial=0)>1e-12:
                raise ContactPhaseExit('Contact velocity requires impact or release')
            key=active.tobytes()
            if key not in self.gauss_cache:
                oracle=FrozenContactOracle(self,active,self.u,self.v,self.target)
                self.gauss_cache[key]=DrivenGaussStepper(oracle,self.dt)
                if len(self.gauss_cache)>4:self.gauss_cache.popitem(last=False)
            s=self.gauss_cache[key];o=s.oracle
            yr=np.asarray(o.T.T@(self.M*(self.u-o.u0))).ravel()/o.mass
            vr=np.asarray(o.T.T@(self.M*self.v)).ravel()/o.mass
            mapped_u=o.u0+o.T@yr;mapped_v=o.T@vr
            if abs(mapped_u-self.u).max()>1e-15 or abs(mapped_v-self.v).max()>1e-12:
                raise ContactPhaseExit('No roundoff-level compatible reduced state')
            s.z=np.r_[o.omega*yr,vr,1.];s.target=self.target
            result=s.step(target,rate_start,rate_end)
        except ContactPhaseExit:
            row=super().step(target)
            row.update(gauss_step=0,contact_phase_fallback=1,gauss_mapping_position_error_m=0.,gauss_mapping_velocity_error_m_s=0.,gauss_impulse_residual_Ns=0.)
            return row
        old=self.u.copy();vold=self.v.copy();u=result.pop('u');v=result.pop('v')
        # Include roundoff from the reduced-coordinate change of representation
        # in the physical energy ledger; it is not silently discarded.
        correction_K=float(.5*np.sum(self.M*(mapped_v*mapped_v-vold*vold)))
        correction_U=float(.5*(mapped_u@(self.K@mapped_u)-old@(self.K@old)))
        correction_spring=float(.5*self.k*((mapped_u[-1]-self.target)**2-(old[-1]-self.target)**2))
        result['kinetic_change_J']+=correction_K;result['elastic_change_J']+=correction_U
        result['drive_spring_change_J']+=correction_spring
        result['mechanical_work_residual_J']+=correction_K+correction_U
        result['full_work_residual_J']+=correction_K+correction_U+correction_spring
        u1,v1,lam1,u2,v2,lam2=s.last_stages;h=self.dt;motion=.5*(v1+v2)
        contact_impulse=.5*h*(self.B.T@(lam1+lam2))
        drag=-h*(self.Dsoft@motion[:-1]).reshape(-1,3)
        actual_change=self.mass[:,None]*(v[:-1]-vold[:-1]).reshape(-1,3)
        contact_nodes=contact_impulse[:-1].reshape(-1,3)
        angular_delta=np.cross(self.rest,actual_change).sum(0)
        angular_ext=np.cross(self.rest,contact_nodes+drag).sum(0)
        angular_residual=float(abs(angular_delta-angular_ext).max())
        impulse_residual=result['impulse_residual_Ns']+float(abs(self.M*(mapped_v-vold)).max())
        position_residual=float(abs(u-old-h*motion).max())
        penetration=0.;minimum_J=float('inf')
        for state in (u1,u2,u):
            x=self.rest+state[:-1].reshape(-1,3);xt=x[self.tets];rt=self.rest[self.tets]
            jac=np.linalg.det((xt[:,1:]-xt[:,:1]).transpose(0,2,1))/(6*self.vol)
            jac*=np.sign(np.linalg.det((rt[:,1:]-rt[:,:1]).transpose(0,2,1)))
            minimum_J=min(minimum_J,float(jac.min()))
            penetration=max(penetration,float(self.table_z-x[:,2].min()),float(x[:,2].max()-self.plate_bottom-state[-1]))
        if not np.isfinite(u).all() or minimum_J<=0 or penetration>1e-9:
            raise RuntimeError('Hybrid Gauss left admissible planar geometry')
        if abs(result['full_work_residual_J'])>1e-9 or impulse_residual>1e-10 or angular_residual>1e-10 or position_residual>1e-10:
            raise RuntimeError('Hybrid Gauss mapped work/impulse/kinematics failed')
        nodal=v[:-1].reshape(-1,3);com=(self.mass[:,None]*nodal).sum(0)/self.mass.sum()
        closed=(self.g0+self.B@u)<=max(5e-17,2*max(5e-18,min(1e-11,h*1e-9)))
        inward=float(max(0.,-np.min((self.B@v)[closed],initial=0.)))
        if inward>1e-9:raise RuntimeError('Hybrid Gauss inward endpoint velocity')
        natural=0.
        for us,lam in ((u1,lam1),(u2,lam2)):
            gs=self.g0+self.B@us;sd=np.diag(self.S)
            natural=max(natural,float(abs((lam-np.maximum(0.,lam-gs/sd))*sd).max()))
        self.u=u;self.v=v;self.target=target;self.step_index+=1
        x=self.rest+u[:-1].reshape(-1,3)
        row=dict(step=self.step_index,time_s=self.step_index*h,tool_q_m=float(u[-1]),tool_v_m_s=float(v[-1]),
            strain=float(1-np.ptp([x[self.bottom,2].mean(),x[self.top,2].mean()])/np.ptp(self.rest[:,2])),
            max_nodal_speed_m_s=float(np.linalg.norm(nodal,axis=1).max()),max_penetration_m=max(0.,penetration),min_J=minimum_J,
            com_vx_m_s=float(com[0]),com_vy_m_s=float(com[1]),com_vz_m_s=float(com[2]),
            max_translation_removed_speed_m_s=float(np.linalg.norm(nodal-com,axis=1).max()),
            internal_force_sum_N=float(np.linalg.norm((self.Ksoft@(.5*(u1[:-1]+u2[:-1]))).reshape(-1,3).sum(0))),
            force_residual_N=result['max_force_residual_N'],natural_residual_m=natural,iterations=0,active=int(active.sum()),
            integration_loss_J=0.,total_integration_loss_J=0.,midpoint_step=0,contact_activation_fallback=0,
            impact_work_J=0.,impact_kinetic_jump_J=0.,impact_work_residual_J=0.,impact_impulse_residual_Ns=0.,gauss_impulse_residual_Ns=impulse_residual,
            impact_iterations=0,impact_top_impulse_Ns=0.,impact_bottom_impulse_Ns=0.,closed_inward_speed_m_s=inward,
            position_integrated_velocity_residual_m=position_residual,peak_preimpact_nodal_speed_m_s=float(np.linalg.norm(nodal,axis=1).max()),
            linearized_angular_impulse_residual_Nms=angular_residual,soft_drag_impulse_z_Ns=float(drag[:,2].sum()),
            gauss_step=1,contact_phase_fallback=0,gauss_mapping_position_error_m=float(abs(mapped_u-old).max()),
            gauss_mapping_velocity_error_m_s=float(abs(mapped_v-vold).max()))
        for k in ('kinetic_change_J','elastic_change_J','drive_work_J','contact_work_J','material_drag_dissipation_J',
            'drive_dissipation_J','drive_spring_change_J','drive_target_work_J','mechanical_work_residual_J','full_work_residual_J',
            'drive_force_N','reaction_N','bottom_reaction_N'):row[k]=result[k]
        for axis,index in zip('xyz',range(3)):
            row['contact_angular_impulse_'+axis+'_Nms']=float(np.cross(self.rest,contact_nodes).sum(0)[index])
        return row
