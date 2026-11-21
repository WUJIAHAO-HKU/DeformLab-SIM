"""Midpoint on smooth contact phases; backward Euler on contact activation.

The switch is a numerical treatment of nonsmooth events, not a constitutive
change. Both branches solve the same material and drive, with separate energy
identities. Endpoint inelastic impulses are inherited and always audited.
"""
import numpy as np
from scipy.sparse import diags
from scipy.sparse.linalg import splu
from .coupled_coupon_reference import CoupledCoupon,ImpactCoupledCoupon,ContactQP

class HybridMidpointCoupon(ImpactCoupledCoupon):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs);h=self.dt
        H=diags(2*self.M/h**2)+self.D/h+.5*self.Ktotal
        self.mid_factor=splu(H.tocsc());self.mid_X=self.mid_factor.solve(self.B.T)
        S=self.B@self.mid_X;self.mid_qp=ContactQP(.5*(S+S.T))

    def integrated_step(self,target):
        if not np.isfinite(target):raise ValueError('Nonfinite drive target')
        h=self.dt;old=self.u.copy();v0=self.v.copy();target0=self.target;target_mid=.5*(target+target0)
        rhs=2*self.M*v0/h-self.Ktotal@old;rhs[-1]+=self.k*target_mid
        free=self.mid_factor.solve(rhs);gap0=self.g0+self.B@old
        tol=max(5e-18,min(1e-11,h*1e-9));lam,gap,qp=self.mid_qp.solve(gap0+self.B@free,tolerance=tol)
        new_contact=(lam>1e-10)&((gap0>max(5e-17,2*tol))|(abs(self.B@v0)>1e-9))
        if np.any(new_contact):
            row=CoupledCoupon.step(self,target)
            row.update(midpoint_step=0,contact_activation_fallback=1)
            return row
        du=free+self.mid_X@lam;u=old+du;motion=du/h;v=2*motion-v0;mid=old+.5*du
        drive=self.k*(target_mid-mid[-1])-self.c*motion[-1]
        applied=np.zeros(len(u));applied[-1]=drive
        force_residual=self.M*(v-v0)/h+self.K@mid+np.r_[self.Dsoft@motion[:-1],0.]-applied-self.B.T@lam
        dK=.5*np.sum(self.M*(v*v-v0*v0));dU=.5*(u@(self.K@u)-old@(self.K@old))
        dw=drive*du[-1];cw=float(lam@(self.B@du));dd=h*float(motion[:-1]@(self.Dsoft@motion[:-1]))
        residual=dK+dU-dw-cw+dd
        spring_delta=.5*self.k*((u[-1]-target)**2-(old[-1]-target0)**2)
        target_work=-self.k*(mid[-1]-target_mid)*(target-target0)
        drive_diss=h*self.c*motion[-1]**2
        full_residual=dK+dU+spring_delta-target_work-cw+dd+drive_diss
        x=self.rest+u[:-1].reshape(-1,3);xt=x[self.tets];rt=self.rest[self.tets]
        J=np.linalg.det((xt[:,1:]-xt[:,:1]).transpose(0,2,1))/(6*self.vol)
        J*=np.sign(np.linalg.det((rt[:,1:]-rt[:,:1]).transpose(0,2,1)))
        penetration=max(0.,float(self.table_z-x[:,2].min()),float(x[:,2].max()-self.plate_bottom-u[-1]))
        if not np.isfinite(u).all() or J.min()<=0 or penetration>1e-9:raise RuntimeError('Midpoint left admissible planar regime')
        if abs(force_residual).max()>1e-6 or abs(residual)>1e-9 or abs(full_residual)>1e-9 or cw>1e-9:
            raise RuntimeError(f'Midpoint balance failed: h={h}, force={abs(force_residual).max()}, work={full_residual}, contact={cw}')
        self.u=u;self.v=v;self.target=target;self.step_index+=1;self.position_contact_tolerance=tol
        self.last_contact_force=lam.copy();self.last_integrated_velocity=v.copy();self.last_motion_velocity=motion.copy();self.last_position_increment=u-old
        nodal=v[:-1].reshape(-1,3);com=(self.mass[:,None]*nodal).sum(0)/self.mass.sum()
        return dict(step=self.step_index,time_s=self.step_index*h,tool_q_m=float(u[-1]),tool_v_m_s=float(v[-1]),
            drive_force_N=float(drive),reaction_N=float(lam[:len(self.top)].sum()),bottom_reaction_N=float(lam[len(self.top):].sum()),
            strain=float(1-(x[self.top,2].mean()-x[self.bottom,2].mean())/np.ptp(self.rest[:,2])),
            max_nodal_speed_m_s=float(np.linalg.norm(nodal,axis=1).max()),max_penetration_m=penetration,min_J=float(J.min()),
            com_vx_m_s=float(com[0]),com_vy_m_s=float(com[1]),com_vz_m_s=float(com[2]),
            max_translation_removed_speed_m_s=float(np.linalg.norm(nodal-com,axis=1).max()),
            internal_force_sum_N=float(np.linalg.norm((self.Ksoft@mid[:-1]).reshape(-1,3).sum(0))),
            force_residual_N=float(abs(force_residual).max()),kinetic_change_J=float(dK),elastic_change_J=float(dU),
            drive_work_J=float(dw),contact_work_J=cw,material_drag_dissipation_J=dd,integration_loss_J=0.,
            mechanical_work_residual_J=float(residual),drive_target_work_J=float(target_work),drive_spring_change_J=float(spring_delta),
            drive_dissipation_J=float(drive_diss),total_integration_loss_J=0.,full_work_residual_J=float(full_residual),
            midpoint_step=1,contact_activation_fallback=0,**qp)
