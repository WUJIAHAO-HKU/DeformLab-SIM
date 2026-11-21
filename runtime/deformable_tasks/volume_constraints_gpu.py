"""GPU colored nonlinear projection of J >= floor with mass-weighted updates.

The candidate adds a hard admissibility inequality to the native step. It is a
split solver, not an inversion-free continuous trajectory certificate. Record
native pre-projection inversions and the projection's energy/impulse costs.
"""
import warp as wp

@wp.kernel
def begin(p:wp.array(dtype=wp.vec3),x:wp.array(dtype=wp.vec3d)):
    i=wp.tid();x[i]=wp.vec3d(p[i])

@wp.kernel
def project(x:wp.array(dtype=wp.vec3d),tets:wp.array(dtype=wp.vec4i),
            order:wp.array(dtype=wp.int32),imass:wp.array(dtype=wp.float64),
            rest6:wp.array(dtype=wp.float64),floor:wp.float64,stats:wp.array(dtype=wp.float64)):
    k=order[wp.tid()];ids=tets[k]
    p0=x[ids[0]];a=x[ids[1]]-p0;b=x[ids[2]]-p0;c=x[ids[3]]-p0
    ratio=wp.dot(a,wp.cross(b,c))/rest6[k]
    if ratio<floor:
        g1=wp.cross(b,c)/rest6[k];g2=wp.cross(c,a)/rest6[k];g3=wp.cross(a,b)/rest6[k];g0=-g1-g2-g3
        w0=imass[ids[0]];w1=imass[ids[1]];w2=imass[ids[2]];w3=imass[ids[3]]
        denom=w0*wp.dot(g0,g0)+w1*wp.dot(g1,g1)+w2*wp.dot(g2,g2)+w3*wp.dot(g3,g3)
        if denom>wp.float64(1.e-24):
            lam=(floor-ratio)/denom
            x[ids[0]]=p0+w0*lam*g0
            x[ids[1]]=x[ids[1]]+w1*lam*g1
            x[ids[2]]=x[ids[2]]+w2*lam*g2
            x[ids[3]]=x[ids[3]]+w3*lam*g3
            wp.atomic_add(stats,0,wp.float64(1.0))

@wp.kernel
def measure(p:wp.array(dtype=wp.vec3),tets:wp.array(dtype=wp.vec4i),rest6:wp.array(dtype=wp.float64),
            stats:wp.array(dtype=wp.float64),slot:int):
    k=wp.tid();ids=tets[k];p0=wp.vec3d(p[ids[0]])
    a=wp.vec3d(p[ids[1]])-p0;b=wp.vec3d(p[ids[2]])-p0;c=wp.vec3d(p[ids[3]])-p0
    ratio=wp.dot(a,wp.cross(b,c))/rest6[k]
    wp.atomic_min(stats,slot,ratio)
    if slot==1 and ratio<=wp.float64(0.0):wp.atomic_add(stats,2,wp.float64(1.0))

@wp.kernel
def publish(p:wp.array(dtype=wp.vec3),v:wp.array(dtype=wp.vec3),x:wp.array(dtype=wp.vec3d),
            imass:wp.array(dtype=wp.float64),invdt:wp.float64,stats:wp.array(dtype=wp.float64),
            impulse:wp.array(dtype=wp.float64)):
    i=wp.tid();oldp=p[i];oldv=v[i];dx=x[i]-wp.vec3d(oldp);nextv=wp.vec3(wp.vec3d(oldv)+dx*invdt)
    p[i]=wp.vec3(x[i]);v[i]=nextv
    mass=wp.float64(1.0)/imass[i];j=mass*(wp.vec3d(nextv)-wp.vec3d(oldv))
    for axis in range(3):wp.atomic_add(impulse,axis,j[axis])
    work=wp.float64(0.5)*mass*(wp.dot(wp.vec3d(nextv),wp.vec3d(nextv))-wp.dot(wp.vec3d(oldv),wp.vec3d(oldv)))
    wp.atomic_add(stats,3,work)
    wp.atomic_max(stats,4,wp.length(dx))
    wp.atomic_max(stats,5,wp.length(wp.vec3d(p[i])-wp.vec3d(oldp)-dx))

@wp.kernel
def impulse_check(impulse:wp.array(dtype=wp.float64),stats:wp.array(dtype=wp.float64)):
    norm=wp.sqrt(impulse[0]*impulse[0]+impulse[1]*impulse[1]+impulse[2]*impulse[2])
    wp.atomic_max(stats,6,norm)
