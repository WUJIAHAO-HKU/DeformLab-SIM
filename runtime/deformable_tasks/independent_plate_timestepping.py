"""Independent small-step planar-contact reference.

Explicit elastic force, backward-Euler drive, position-level unilateral impulses.
First order: numerical dissipation and material quadrature defects are reported,
not hidden or confused with physical damping. This is not the native backend.
No production contact QP, discrete material gradient, or event locator is used.
"""
import numpy as np


def upper_impulses(mass, free_velocity, effective_plate_mass, free_plate_velocity, gap, h):
    """Solve diagonal-plus-rank-one unilateral contact by sorted scalar means."""
    mass=np.asarray(mass,float);speed=np.asarray(free_velocity,float)-np.asarray(gap,float)/h
    mean=float(free_plate_velocity);weighted=effective_plate_mass*mean;total=effective_plate_mass
    bound=[]
    for i in np.argsort(-speed):
        if speed[i]>mean:
            bound.append(i);weighted+=mass[i]*speed[i];total+=mass[i];mean=weighted/total
    impulse=np.zeros(len(mass))
    if bound:impulse[bound]=mass[bound]*(speed[bound]-mean)
    return impulse,float(mean)


def advance(model,mass,x,v,q,w,gaps,bottom,top,target0,target1,h,
            plate_mass=.2,stiffness=1e6,damping=4000.,energy_gradient=None):
    if h<=0:raise ValueError('Positive step required')
    mass=np.asarray(mass);bottom=np.asarray(bottom,int);top=np.asarray(top,int)
    U,g=model.energy_gradient(x) if energy_gradient is None else energy_gradient
    effective_mass=plate_mass+h*damping+h*h*stiffness
    vf=v-h*g/mass[:,None]
    wf=(plate_mass*w-h*stiffness*(q-target1))/effective_mass
    jl=np.maximum(-mass[bottom]*(vf[bottom,2]+gaps[:len(bottom)]/h),0.)
    ju,wn=upper_impulses(mass[top],vf[top,2],effective_mass,wf,gaps[len(bottom):],h)
    vn=vf.copy();vn[bottom,2]+=jl/mass[bottom];vn[top,2]-=ju/mass[top]
    dx=h*vn;dq=h*wn;xn=x+dx;qn=q+dq
    gn=gaps+h*np.r_[vn[bottom,2],wn-vn[top,2]]
    Un,gradient_next=model.energy_gradient(xn)
    drive_impulse=-h*(stiffness*(qn-target1)+damping*wn)
    momentum=np.sum(mass[:,None]*(vn-v),axis=0)+[0.,0.,plate_mass*(wn-w)]
    momentum-=np.array([0.,0.,jl.sum()+drive_impulse])
    contact_work=float(jl@vn[bottom,2]+ju@(wn-vn[top,2]))
    numerical=float(Un-U-np.sum(g*dx)-.5*np.sum(mass[:,None]*(vn-v)**2)
                    -.5*plate_mass*(wn-w)**2-.5*stiffness*(dq-(target1-target0))**2)
    drive_loss=h*damping*wn**2
    target_work=-stiffness*(qn-target1)*(target1-target0)
    delta_energy=float(Un-U+.5*np.sum(mass[:,None]*(vn**2-v**2))
        +.5*plate_mass*(wn**2-w**2)+.5*stiffness*((qn-target1)**2-(q-target0)**2))
    residual=delta_energy+drive_loss-contact_work-target_work-numerical
    j=np.r_[jl,ju]
    if gn.min(initial=0)<-1e-12 or j.min(initial=0)<-1e-12:
        raise RuntimeError('Independent contact feasibility failed')
    row=dict(lower_impulses_Ns=jl,upper_impulses_Ns=ju,
             numerical_energy_change_J=numerical,contact_work_J=contact_work,
             drive_loss_J=drive_loss,target_work_J=target_work,energy_delta_J=delta_energy,
             work_identity_residual_J=residual,momentum_residual_Ns=float(np.linalg.norm(momentum)),
             max_gap_complementarity_Nsm=float(np.max(abs(j*gn),initial=0.)))
    return xn,vn,qn,float(wn),gn,(Un,gradient_next),row
