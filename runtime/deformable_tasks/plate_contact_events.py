"""Instantaneous contact forces and plastic impact for the finite plate reference.

Planar fixed normals, one translating rigid plate. No position projection. These
operators do not by themselves locate contact events or advance the trajectory.
"""
import numpy as np
from .coupled_coupon_reference import ContactQP


def instantaneous(model,mass,x,v,q,w,bottom,top,target,plate_mass=.2,stiffness=1e6,damping=4000.):
    bottom=np.asarray(bottom,int);top=np.asarray(top,int);mass=np.asarray(mass,float)
    _,g=model.energy_gradient(x)
    acceleration=-g/mass[:,None]
    drive=stiffness*(q-target)+damping*w
    plate_acceleration=-(drive+g[top,2].sum())/(plate_mass+mass[top].sum())
    acceleration[bottom,2]=0.;acceleration[top,2]=plate_acceleration
    r=mass[:,None]*acceleration+g
    return acceleration,float(plate_acceleration),r[bottom,2],-r[top,2]


def acceleration_complementarity(model,mass,x,q,w,bottom,top,target,plate_mass=.2,stiffness=1e6,damping=4000.):
    """Choose compressive resting contacts using unilateral acceleration KKT."""
    bottom=np.asarray(bottom,int);top=np.asarray(top,int);mass=np.asarray(mass,float)
    if len(bottom)+len(top)==0:return np.empty(0),np.empty(0)
    _,g=model.energy_gradient(x);a=-g/mass[:,None]
    aq=-(stiffness*(q-target)+damping*w)/plate_mass
    free=np.r_[a[bottom,2],aq-a[top,2]]
    S=np.diag(np.r_[1/mass[bottom],1/mass[top]])
    S[len(bottom):,len(bottom):]+=1/plate_mass
    force,normal_acceleration,_=ContactQP(S).solve(free,tolerance=1e-10)
    if force.min(initial=0)<-1e-9 or normal_acceleration.min(initial=0)<-1e-8:
        raise RuntimeError('Resting contact acceleration KKT failed')
    return force,normal_acceleration


def plastic_impact(mass,x,v,q,w,bottom,top,table,plate_mass=.2,gap_tolerance=1e-12):
    mass=np.asarray(mass,float);x=np.asarray(x,float);v=np.asarray(v,float)
    bottom=np.asarray(bottom,int);top=np.asarray(top,int)
    if len(np.intersect1d(bottom,top)):raise ValueError('A node cannot touch both planes')
    gaps=np.r_[x[bottom,2]-table,q-x[top,2]]
    if np.max(abs(gaps),initial=0)>gap_tolerance:raise ValueError('Impact requires closed contacts')
    n=len(bottom)+len(top)
    if n==0:raise ValueError('No impact contacts')
    S=np.diag(np.r_[1/mass[bottom],1/mass[top]])
    S[len(bottom):,len(bottom):]+=1/plate_mass
    before=np.r_[v[bottom,2],w-v[top,2]]
    # Resolve closing velocities below the event controller's resting threshold;
    # a looser QP tolerance can return zero impulse and repeat the same event.
    closing=float(np.max(-before,initial=0.))
    qp_tolerance=min(1e-15,.1*closing) if closing>0 else 1e-15
    impulse,after,qp=ContactQP(S).solve(before,tolerance=qp_tolerance)
    new=v.copy();new[bottom,2]+=impulse[:len(bottom)]/mass[bottom]
    new[top,2]-=impulse[len(bottom):]/mass[top]
    w1=w+impulse[len(bottom):].sum()/plate_mass
    # Publish a single common velocity for binding rows. Separate floating
    # evaluations of v+J/m and w+sum(J)/M otherwise leave a tiny relative speed
    # which gets amplified into force by a subsequent very short event step.
    roundoff_bound=32*np.finfo(float).eps*max(float(np.max(abs(v),initial=0.)),abs(w),abs(w1),np.finfo(float).tiny)
    unrounded=new.copy();unrounded_w=w1
    bind_lower=(impulse[:len(bottom)]>0) | (abs(new[bottom,2])<=roundoff_bound)
    bind_upper=(impulse[len(bottom):]>0) | (abs(w1-new[top,2])<=roundoff_bound)
    new[bottom[bind_lower],2]=0.
    if bind_upper.any():
        nodes=top[bind_upper]
        w1=float((plate_mass*w1+np.sum(mass[nodes]*new[nodes,2]))/(plate_mass+mass[nodes].sum()))
        new[nodes,2]=w1
    publication=max(float(np.max(abs(new-unrounded),initial=0.)),abs(w1-unrounded_w))
    if publication>roundoff_bound:raise RuntimeError('Binding velocity publication exceeds roundoff bound')
    actual_after=np.r_[new[bottom,2],w1-new[top,2]]
    soft_impulse=mass[:,None]*(new-v)
    dP=soft_impulse.sum(0)+[0,0,plate_mass*(w1-w)]
    external=np.array([0,0,impulse[:len(bottom)].sum()])
    deltaK=.5*np.sum(mass[:,None]*(new**2-v**2))+.5*plate_mass*(w1**2-w**2)
    impulse_work=.5*float(impulse@(before+actual_after))
    dL=np.sum(np.cross(x,mass[:,None]*new)-np.cross(x,mass[:,None]*v),0)
    angular=np.sum(np.cross(x,soft_impulse),0)
    report=dict(lower_impulses_Ns=impulse[:len(bottom)].tolist(),upper_impulses_Ns=impulse[len(bottom):].tolist(),
        kinetic_change_J=float(deltaK),dissipated_J=float(-deltaK),impulse_work_J=impulse_work,
        work_residual_J=float(deltaK-impulse_work),external_momentum_residual_Ns=float(np.linalg.norm(dP-external)),
        soft_angular_impulse_residual_Nms=float(np.linalg.norm(dL-angular)),
        min_impulse_Ns=float(impulse.min()),min_post_normal_velocity_m_s=float(actual_after.min()),
        complementarity_Nm_s=float(np.max(abs(impulse*actual_after))),
        velocity_map_residual_m_s=float(np.max(abs(actual_after-after))),
        position_change_m=0.,binding_velocity_publication_m_s=publication,
        binding_velocity_roundoff_bound_m_s=roundoff_bound,
        velocity_publication_energy_J=float(.5*np.sum(mass[:,None]*(new**2-unrounded**2))+.5*plate_mass*(w1**2-unrounded_w**2)),
        velocity_publication_momentum_Ns=float(np.linalg.norm(np.sum(mass[:,None]*(new-unrounded),0)+[0,0,plate_mass*(w1-unrounded_w)])),
        qp_velocity_tolerance_m_s=qp_tolerance,qp_iterations=qp['iterations'])
    if (report['min_impulse_Ns']<-1e-12 or report['min_post_normal_velocity_m_s']<-1e-10
        or report['complementarity_Nm_s']>1e-10 or report['dissipated_J']<-1e-12
        or abs(report['work_residual_J'])>1e-10 or report['external_momentum_residual_Ns']>1e-10):
        raise RuntimeError('Plastic impact invariant failed: '+repr(report))
    return new,float(w1),report
