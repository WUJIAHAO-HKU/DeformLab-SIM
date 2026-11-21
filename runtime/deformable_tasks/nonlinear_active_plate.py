"""Finite-strain material/one-DOF plate/drive solve for a fixed contact set.

Frictionless, planar, already closed contacts only. This constrained reference
does not detect impacts or release contacts. Negative reactions mark an invalid
unilateral-contact state; they are never clipped or accepted as physical contact.
"""
import numpy as np
from scipy.sparse.linalg import LinearOperator, gmres
from .corotated_discrete_work import discrete_material_gradient
from .corotated_discrete_tangent import DiscreteMaterialTangent


def step(model,mass,x,v,q,w,bottom,top,table,target0,target1,h,
         plate_mass=.2,stiffness=1e6,damping=4000.,tolerance_N=1e-8,max_iterations=30):
    x=np.asarray(x,float);v=np.asarray(v,float);mass=np.asarray(mass,float)
    bottom=np.asarray(bottom,int);top=np.asarray(top,int)
    if (mass.shape!=(len(x),) or np.any(mass<=0) or not np.isfinite(mass).all()
        or not np.isfinite([q,w,table,target0,target1,h,plate_mass,stiffness,damping]).all()
        or h<=0 or plate_mass<=0 or stiffness<0 or damping<0):raise ValueError('Invalid plate parameters')
    if (len(np.intersect1d(bottom,top)) or any(ids.ndim!=1 or len(np.unique(ids))!=len(ids) or
        np.any(ids<0) or np.any(ids>=len(x)) for ids in (bottom,top))):raise ValueError('Invalid fixed contact sets')
    maxabs=lambda a:float(np.max(abs(a),initial=0.))
    if max(maxabs(x[bottom,2]-table),maxabs(x[top,2]-q))>1e-12:raise ValueError('Initial contacts must be closed')
    if max(maxabs(v[bottom,2]),maxabs(v[top,2]-w))>1e-10:raise ValueError('Initial contact velocities must match')
    model.kinematics(x)
    free=np.ones(x.size,bool);free[3*bottom+2]=False;free[3*top+2]=False
    indices=np.flatnonzero(free);size=len(indices)+1
    def linear_expand(z):
        d=np.zeros(x.size);d[indices]=z[:-1];d[3*top+2]=z[-1]
        return d.reshape(x.shape),z[-1]
    def acceleration(z):
        a,aq=linear_expand(z)
        a[bottom,2]=-2*v[bottom,2]/h
        a[top,2]+=2*(w-v[top,2])/h
        return a,aq
    def expand(z):
        a,aq=acceleration(z);dx=h*v+.5*h*h*a;dq=h*w+.5*h*h*aq
        dx[bottom,2]=0.;dx[top,2]=dq
        return dx,dq
    def reduce(r,rq):return np.r_[r.ravel()[indices],rq+r[top,2].sum()]
    m=mass[:,None]
    def evaluate(z):
        dx,dq=expand(z);g,info=discrete_material_gradient(model,x,x+dx)
        a,aq=acceleration(z);r=m*a+g
        drive=stiffness*(q+.5*dq-.5*(target0+target1))+damping*dq/h
        rq=plate_mass*aq+drive
        return reduce(r,rq),r,rq,info
    z=np.zeros(size);history=[];linear=[]
    diagonal=np.r_[np.repeat(mass,3)[indices],
                   plate_mass+mass[top].sum()+.25*stiffness*h*h+.5*damping*h]
    pre=LinearOperator((size,size),matvec=lambda d:d/diagonal,dtype=float)
    for iteration in range(max_iterations+1):
        residual,r,rq,info=evaluate(z);error=float(abs(residual).max());history.append(error)
        if error<tolerance_N:break
        if iteration==max_iterations:raise RuntimeError('Coupled plate Newton failed: '+repr(history))
        dx,_=expand(z);tangent=DiscreteMaterialTangent(model,x,x+dx)
        def product(direction):
            d,aq=linear_expand(direction)
            return reduce(m*d+.5*h*h*tangent.product(d),
                          (plate_mass+.25*stiffness*h*h+.5*damping*h)*aq)
        operator=LinearOperator((size,size),matvec=product,dtype=float);counts=[]
        dz,status=gmres(operator,-residual,M=pre,rtol=1e-10,atol=1e-12,
            restart=min(size,100),maxiter=100,callback=counts.append,callback_type='pr_norm')
        linear.append(len(counts))
        if status!=0 or np.linalg.norm(product(dz)+residual)>max(1e-11,1e-9*np.linalg.norm(residual)):
            raise RuntimeError('Coupled plate GMRES failed: '+str(status))
        alpha=1.
        for _ in range(30):
            trial=z+alpha*dz
            try:rt,_,_,_=evaluate(trial)
            except ValueError:alpha*=.5;continue
            if np.linalg.norm(rt)<np.linalg.norm(residual):z=trial;break
            alpha*=.5
        else:raise RuntimeError('Coupled plate positive-volume line search failed')
    dx,dq=expand(z);a,aq=acceleration(z);y=x+dx;new=v+h*a;q1=q+dq;w1=w+h*aq
    lower=r[bottom,2];upper=-r[top,2]
    U0,_=model.energy_gradient(x);U1,_=model.energy_gradient(y)
    kinetic=.5*np.sum(m*(new**2-v**2))+.5*plate_mass*(w1**2-w**2)
    spring=.5*stiffness*((q1-target1)**2-(q-target0)**2)
    target_work=-stiffness*(.5*(q+q1)-.5*(target0+target1))*(target1-target0)
    dissipated=damping*dq*dq/h
    soft_contact=np.zeros_like(x);soft_contact[bottom,2]=lower;soft_contact[top,2]=-upper
    contact_work=float(np.sum(soft_contact*dx)+upper.sum()*dq)
    drive=stiffness*(.5*(q+q1)-.5*(target0+target1))+damping*dq/h
    dP=np.sum(m*(new-v),0)+np.array([0,0,plate_mass*(w1-w)])
    external=np.array([0,0,h*(lower.sum()-drive)])
    dL=np.sum(np.cross(y,m*new)-np.cross(x,m*v),0)
    torque=h*np.sum(np.cross(.5*(x+y),soft_contact),0)
    info.update(newton_iterations=iteration,linear_iterations=linear,force_residual_N=error,residual_history_N=history,
        lower_reactions_N=lower.tolist(),upper_reactions_N=upper.tolist(),
        min_reaction_N=float(np.min(np.r_[lower,upper],initial=np.inf)) if len(lower)+len(upper) else 0.,
        unilateral_admissible=bool(np.min(np.r_[lower,upper],initial=0.)>=-1e-8),
        material_energy_change_J=U1-U0,kinetic_energy_change_J=float(kinetic),spring_energy_change_J=float(spring),
        target_work_J=float(target_work),drive_dissipation_J=float(dissipated),contact_work_J=contact_work,
        full_work_residual_J=float(kinetic+U1-U0+spring+dissipated-target_work-contact_work),
        external_momentum_residual_Ns=float(np.linalg.norm(dP-external)),
        soft_angular_impulse_residual_Nms=float(np.linalg.norm(dL-torque)),
        position_constraint_m=max(maxabs(y[bottom,2]-table),maxabs(y[top,2]-q1)),
        velocity_constraint_m_s=max(maxabs(new[bottom,2]),maxabs(new[top,2]-w1)),
        min_J=float(model.kinematics(y)[1].min()))
    return y,new,q1,w1,info
