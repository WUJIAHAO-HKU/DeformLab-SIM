"""Instantaneous planar-contact forces using the evolving solver's damping.

These operators include the original objective gamma and mass-proportional drag.
They do not locate events or advance time. Plastic impact remains a separate
velocity impulse, since finite smooth forces do not supply an impulsive load.
"""
import numpy as np
from .coupled_coupon_reference import ContactQP
from .plate_contact_events import plastic_impact


def resistance(model,mass,x,v,material_gamma,linear_damping):
    mass,x,v = np.asarray(mass,float),np.asarray(x,float),np.asarray(v,float)
    if (x.shape != model.rest.shape or v.shape != x.shape or mass.shape != (len(x),)
            or not all(np.isfinite(a).all() for a in (mass,x,v)) or np.any(mass<=0)
            or not np.isfinite([material_gamma,linear_damping]).all()
            or min(material_gamma,linear_damping)<0):
        raise ValueError('Finite states, positive mass and nonnegative damping required')
    _,elastic = model.energy_gradient(x)
    return elastic+model.directional_damping(x,v,material_gamma)+linear_damping*mass[:,None]*v


def _contacts(mass,bottom,top,q,w,target,plate_mass,stiffness,damping):
    bottom,top = np.asarray(bottom,int),np.asarray(top,int)
    if (not np.isfinite([q,w,target,plate_mass,stiffness,damping]).all()
            or plate_mass<=0 or min(stiffness,damping)<0
            or len(np.intersect1d(bottom,top))
            or any(a.ndim!=1 or len(np.unique(a))!=len(a) or np.any(a<0) or np.any(a>=len(mass))
                   for a in (bottom,top))):
        raise ValueError('Valid plate and disjoint unique contact sets required')
    return bottom,top


def instantaneous(model,mass,x,v,q,w,bottom,top,target,plate_mass=.2,stiffness=1e6,
                  damping=4000.,material_gamma=.05,linear_damping=.1):
    mass = np.asarray(mass,float)
    bottom,top = _contacts(mass,bottom,top,q,w,target,plate_mass,stiffness,damping)
    force = resistance(model,mass,x,v,material_gamma,linear_damping)
    acceleration = -force/mass[:,None]
    drive = stiffness*(q-target)+damping*w
    aq = -(drive+force[top,2].sum())/(plate_mass+mass[top].sum())
    acceleration[bottom,2] = 0.
    acceleration[top,2] = aq
    residual = mass[:,None]*acceleration+force
    return acceleration,float(aq),residual[bottom,2],-residual[top,2]


def acceleration_complementarity(model,mass,x,v,q,w,bottom,top,target,plate_mass=.2,
                                 stiffness=1e6,damping=4000.,material_gamma=.05,linear_damping=.1):
    mass = np.asarray(mass,float)
    bottom,top = _contacts(mass,bottom,top,q,w,target,plate_mass,stiffness,damping)
    force = resistance(model,mass,x,v,material_gamma,linear_damping)
    if len(bottom)+len(top)==0:
        return np.empty(0),np.empty(0)
    acceleration = -force/mass[:,None]
    aq = -(stiffness*(q-target)+damping*w)/plate_mass
    free = np.r_[acceleration[bottom,2],aq-acceleration[top,2]]
    mobility = np.diag(np.r_[1/mass[bottom],1/mass[top]])
    mobility[len(bottom):,len(bottom):] += 1/plate_mass
    reactions,normal,_ = ContactQP(mobility).solve(free,tolerance=1e-10)
    if (reactions.min(initial=0)<-1e-9 or normal.min(initial=0)<-1e-8
            or np.max(abs(reactions*normal),initial=0)>1e-8):
        raise RuntimeError('Damped resting contact acceleration KKT failed')
    return reactions,normal
