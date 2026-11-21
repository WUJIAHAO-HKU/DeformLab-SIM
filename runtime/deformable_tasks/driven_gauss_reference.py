"""Gauss collocation with a consistently integrated prescribed drive target.

Each step uses the cubic Hermite command determined by endpoint positions and
rates. The command is a kinematic state in the SAME Runge--Kutta tableau as the
body. This closes quadratic spring energy using independently evaluated target
force times target rate; target work is never defined by an energy residual.
Endpoint commands are exact, while a general smooth command is approximated to
fourth order. Fixed planar contacts only; transitions are explicitly rejected.
"""
import numpy as np
from .frozen_gauss_reference import FrozenGaussStepper


class ContactPhaseExit(RuntimeError):
    """The chosen contact set cannot represent this trial step."""


class DrivenGaussStepper(FrozenGaussStepper):
    def __init__(self, oracle, dt):
        super().__init__(oracle,dt)
        self.target=oracle.target
        m=oracle.model;n=len(oracle.mass)
        applied=np.zeros(len(m.M));applied[-1]=m.k
        self.input=np.r_[np.zeros(n),np.asarray(oracle.T.T@applied).ravel()/oracle.mass,0.]

    def inspect_driven(self,z,target):
        o=self.oracle;m=o.model;n=len(o.mass)
        u,v=self.decode(z);dz=o.A@z+self.input*(target-o.target)
        acceleration=o.T@dz[n:2*n]
        applied=np.zeros(len(m.M));applied[-1]=m.k*target
        residual=m.M*acceleration+m.D@v+m.Ktotal@u-applied
        lam=np.r_[-residual[3*m.top+2],residual[3*m.bottom+2]];lam[~o.active]=0.
        gap=m.g0+m.B@u;error=float(abs(residual-m.B.T@lam).max())
        if error>1e-6:raise RuntimeError('Driven Gauss force balance failed')
        # A newly closed point with inward velocity is an impact even when its
        # penetration is smaller than the geometry audit's 1 nm threshold.
        # Hand it to the nonsmooth solver before committing the smooth step.
        closing_tolerance=max(5e-17,2*max(5e-18,min(1e-11,self.dt*1e-9)))
        if np.any((~o.active)&(gap<=closing_tolerance)&((m.B@v)<-1e-9)):
            raise ContactPhaseExit('Newly closed contact requires an inelastic impulse')
        if gap.min()<-1e-9 or lam[o.active].min(initial=0)<-1e-8:
            raise ContactPhaseExit('Driven Gauss requires contact activation or release')
        return u,v,lam,error

    def step(self,target_end,rate_start,rate_end):
        if not np.isfinite([target_end,rate_start,rate_end]).all():raise ValueError('Nonfinite drive command')
        h=self.dt;r=np.sqrt(3)/6;old=self.z.copy();target0=self.target
        def derivative(c):
            return 6*c*(1-c)*(target_end-target0)/h+(3*c*c-4*c+1)*rate_start+(3*c*c-2*c)*rate_end
        p1=derivative(.5-r);p2=derivative(.5+r)
        t1=target0+h*(.25*p1+(.25-r)*p2);t2=target0+h*((.25+r)*p1+.25*p2)
        target_integrated=target0+.5*h*(p1+p2)
        if abs(target_integrated-target_end)>1e-14:raise RuntimeError('Hermite command endpoint inconsistency')
        b1=self.input*(t1-self.oracle.target);b2=self.input*(t2-self.oracle.target)
        total=self.factor.solve(2*(self.oracle.A@old)+b1+b2-h*r*(self.oracle.A@(b2-b1)))
        difference=h*r*(self.oracle.A@total)+b2-b1
        k1=.5*(total-difference);k2=.5*(total+difference)
        new=old+.5*h*total
        stage1=old+h*(.25*k1+(.25-r)*k2);stage2=old+h*((.25+r)*k1+.25*k2)
        u0,v0=self.decode(old);u,v,lam_end,end_residual=self.inspect_driven(new,target_end)
        u1,v1,lam1,r1=self.inspect_driven(stage1,t1);u2,v2,lam2,r2=self.inspect_driven(stage2,t2)
        m=self.oracle.model
        kinetic_change=.5*np.sum(m.M*(v*v-v0*v0))
        elastic_change=.5*(u@(m.K@u)-u0@(m.K@u0))
        force1=m.k*(t1-u1[-1])-m.c*v1[-1];force2=m.k*(t2-u2[-1])-m.c*v2[-1]
        drive_work=.5*h*(force1*v1[-1]+force2*v2[-1])
        contact_work=.5*h*(lam1@(m.B@v1)+lam2@(m.B@v2))
        material_loss=.5*h*(v1[:-1]@(m.Dsoft@v1[:-1])+v2[:-1]@(m.Dsoft@v2[:-1]))
        drive_loss=.5*h*m.c*(v1[-1]**2+v2[-1]**2)
        spring_change=.5*m.k*((u[-1]-target_end)**2-(u0[-1]-target0)**2)
        target_work=-.5*h*m.k*((u1[-1]-t1)*p1+(u2[-1]-t2)*p2)
        body_residual=kinetic_change+elastic_change-drive_work-contact_work+material_loss
        full_residual=kinetic_change+elastic_change+spring_change-target_work-contact_work+material_loss+drive_loss
        applied=np.zeros(len(m.M));applied[-1]=m.k*(t1+t2)
        impulse=.5*h*(applied-m.D@(v1+v2)-m.Ktotal@(u1+u2)+m.B.T@(lam1+lam2))
        impulse_residual=float(abs(m.M*(v-v0)-impulse).max())
        position_residual=float(abs(u-u0-.5*h*(v1+v2)).max())
        if abs(body_residual)>1e-9 or abs(full_residual)>1e-9 or impulse_residual>1e-9 or position_residual>1e-10:
            raise RuntimeError('Driven Gauss work, impulse or kinematics failed')
        self.z=new;self.target=float(target_end);self.step_index+=1
        self.last_stages=(u1,v1,lam1,u2,v2,lam2)
        return dict(u=u,v=v,kinetic_change_J=float(kinetic_change),elastic_change_J=float(elastic_change),
            drive_work_J=float(drive_work),contact_work_J=float(contact_work),material_drag_dissipation_J=float(material_loss),
            drive_dissipation_J=float(drive_loss),drive_spring_change_J=float(spring_change),drive_target_work_J=float(target_work),
            mechanical_work_residual_J=float(body_residual),full_work_residual_J=float(full_residual),
            impulse_residual_Ns=impulse_residual,position_residual_m=position_residual,
            target_endpoint_residual_m=abs(target_integrated-target_end),max_force_residual_N=max(r1,r2,end_residual),
            drive_force_N=float(.5*(force1+force2)),reaction_N=float(.5*(lam1[:len(m.top)].sum()+lam2[:len(m.top)].sum())),
            bottom_reaction_N=float(.5*(lam1[len(m.top):].sum()+lam2[len(m.top):].sum())))
