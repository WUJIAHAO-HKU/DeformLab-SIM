"""Independent homogeneous, free-lateral uniaxial corotated material reference.

Diagonal F=(s,s,l), zero lateral nominal traction, positive stretches. This
constitutive equilibrium is not a contact/dynamic solver or hardware truth.
"""
import math

def uniaxial_corotated(young,poisson,axial_stretch,reference_area=1.):
    values=(young,poisson,axial_stretch,reference_area)
    if not all(math.isfinite(v) for v in values) or young<=0 or not 0<=poisson<.5 or axial_stretch<=0 or reference_area<=0:
        raise ValueError('Positive material, stretch and area required')
    mu=young/(2*(1+poisson));lame=young*poisson/((1+poisson)*(1-2*poisson));l=axial_stretch
    # P_xx=2 mu (s-1) + lambda (s^2 l-1) s l. The unique positive
    # traction-free root lies between 1 and l^(-1/2), also under tension.
    lo=min(1.,1/math.sqrt(l));hi=max(1.,1/math.sqrt(l))
    if lame==0:s=1.
    else:
        for _ in range(80):
            s=.5*(lo+hi)
            f=2*mu*(s-1)+lame*(s*s*l-1)*s*l
            if f>0:hi=s
            else:lo=s
        s=.5*(lo+hi)
    J=s*s*l;P=2*mu*(l-1)+lame*(J-1)*s*s
    lateral=2*mu*(s-1)+lame*(J-1)*s*l
    A=2*mu+lame*l*(3*s*s*l-1);B=lame*s*(2*s*s*l-1)
    tangent=2*mu+lame*s**4-2*B*B/A
    energy=mu*(2*(s-1)**2+(l-1)**2)+.5*lame*(J-1)**2
    return dict(axial_stretch=l,lateral_stretch=s,volume_ratio=J,
        nominal_axial_stress_Pa=P,lateral_nominal_stress_Pa=lateral,
        compression_reaction_N=-P*reference_area,
        minimized_energy_density_J_m3=energy,uniaxial_tangent_Pa=tangent,
        scope='Homogeneous positive-stretch corotated equilibrium with zero lateral traction. Requires independently verified quasistatic, frictionless, nearly homogeneous response for comparison.')
