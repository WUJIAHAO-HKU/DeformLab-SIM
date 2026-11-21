"""Fourth-order Gauss collocation for constant-target frozen planar contacts.

Diagnostic smooth-phase candidate, NOT a general contact-switching backend.
Both collocation stages and endpoint must pass unilateral admission. No silent
fallback, post-step damping, extrapolation or mass/velocity modification.
"""
import numpy as np
from scipy.sparse.linalg import LinearOperator
from .gauss_factor import GaussReducedFactor


class FrozenGaussStepper:
    def __init__(self, oracle, dt):
        if not np.isfinite(dt) or dt<=0: raise ValueError('Require positive time step')
        self.oracle=oracle; self.dt=dt; self.z=oracle.z0.copy(); self.step_index=0
        A=oracle.A
        self.factor=GaussReducedFactor(oracle,dt)
        def right_product(z):
            az=A@z
            return z+.5*dt*az+dt**2/12*(A@az)
        self.right=LinearOperator(A.shape,matvec=right_product,dtype=float)

    def decode(self,z):
        o=self.oracle; n=len(o.mass)
        return o.u0+o.T@(z[:n]/o.omega), o.T@z[n:2*n]

    def inspect(self,z):
        o=self.oracle; m=o.model; n=len(o.mass)
        u,v=self.decode(z); dz=o.A@z; acceleration=o.T@dz[n:2*n]
        applied=np.zeros(len(m.M));applied[-1]=m.k*o.target
        residual=m.M*acceleration+m.D@v+m.Ktotal@u-applied
        lam=np.r_[-residual[3*m.top+2],residual[3*m.bottom+2]];lam[~o.active]=0.
        gap=m.g0+m.B@u
        error=float(abs(residual-m.B.T@lam).max())
        if error>1e-6 or gap.min()<-1e-9 or lam[o.active].min(initial=0)<-1e-8:
            raise RuntimeError('Frozen Gauss phase left sampled unilateral regime')
        return u,v,lam,error

    def step(self):
        h=self.dt;old=self.z.copy();new=self.factor.solve(self.right@old)
        delta=new-old;mean=delta/h;difference=np.sqrt(3)/6*(self.oracle.A@delta)
        k1=mean-difference;k2=mean+difference
        stage1=old+h*(.25*k1+(.25-np.sqrt(3)/6)*k2)
        stage2=old+h*((.25+np.sqrt(3)/6)*k1+.25*k2)
        u0,v0=self.decode(old);u,v,_,end_residual=self.inspect(new)
        u1,v1,lam1,r1=self.inspect(stage1);u2,v2,lam2,r2=self.inspect(stage2)
        m=self.oracle.model;target=self.oracle.target
        energy=lambda a,b:.5*np.sum(m.M*b*b)+.5*a@(m.Ktotal@a)-m.k*target*a[-1]
        energy_change=energy(u,v)-energy(u0,v0)
        dissipation=.5*h*(v1@(m.D@v1)+v2@(m.D@v2))
        applied=np.zeros(len(m.M));applied[-1]=m.k*target
        impulse=.5*h*(2*applied-m.D@(v1+v2)-m.Ktotal@(u1+u2)+m.B.T@(lam1+lam2))
        impulse_residual=float(abs(m.M*(v-v0)-impulse).max())
        residual=float(energy_change+dissipation)
        position_residual=float(abs(u-u0-.5*h*(v1+v2)).max())
        if abs(residual)>1e-9 or impulse_residual>1e-9 or position_residual>1e-10:
            raise RuntimeError('Frozen Gauss energy, impulse or kinematics failed')
        self.z=new;self.step_index+=1
        return dict(u=u,v=v,energy_change_J=float(energy_change),dissipation_J=float(dissipation),
            work_residual_J=residual,impulse_residual_Ns=impulse_residual,
            position_residual_m=position_residual,max_force_residual_N=max(r1,r2,end_residual))
