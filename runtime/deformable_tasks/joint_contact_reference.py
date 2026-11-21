"""Independent dense reference for frozen, frictionless joint inequalities.

Same impulse changes g/h and relative velocity by A*d_lambda. Intersection
uses c=min(g/h,v_rel), and solves lambda>=0, c+A*(lambda-lambda_old)>=0.
This reference includes cross-contact coupling. It is not a FEM time step.
"""
import numpy as np
from scipy.optimize import minimize
try:
    from threadpoolctl import threadpool_limits
except ImportError:
    from contextlib import nullcontext
    def threadpool_limits(**kwargs):return nullcontext()


def solve_joint(A, gap, relative_velocity, h, previous, tolerance=2e-6):
    A=np.asarray(A,float);gap=np.asarray(gap,float);v=np.asarray(relative_velocity,float);old=np.asarray(previous,float)
    if h<=0 or A.shape!=(len(gap),len(gap)) or len(v)!=len(gap) or len(old)!=len(gap):raise ValueError('Invalid dimensions/time step')
    if any(not np.isfinite(x).all() for x in (A,gap,v,old)) or np.any(old<0):raise ValueError('Invalid state')
    if not np.allclose(A,A.T,atol=1e-10,rtol=1e-10) or np.any(np.diag(A)<=0):raise ValueError('Invalid contact operator')
    if not len(gap):return dict(impulse=np.zeros(0),increment=np.zeros(0),accepted=True,projected_residual_m_s=0.)
    c=np.minimum(gap/h,v);scale=np.sqrt(np.diag(A));scaled=A/scale[:,None]/scale[None,:]
    linear=(c-A@old)/scale
    with threadpool_limits(limits=1):
        def fun(x):return .5*x@(scaled@x)+linear@x,scaled@x+linear
        res=minimize(fun,old*scale,jac=True,bounds=[(0,None)]*len(gap),method='L-BFGS-B',
            options=dict(maxiter=10000,maxls=50,ftol=1e-15,gtol=1e-12,maxcor=30))
        # Degenerate/redundant contact rows can stop L-BFGS on relative function
        # reduction while complementarity remains poor. Never accept that flag.
        probe=res.x/scale
        defect=(c+A@(probe-old))
        defect=np.max(abs(probe-np.maximum(0.,probe-defect/np.diag(A)))*np.diag(A))
        if defect>=tolerance:
            res=minimize(fun,res.x,jac=True,bounds=[(0,None)]*len(gap),method='SLSQP',
                options=dict(maxiter=3000,ftol=1e-14))
    lam=res.x/scale;delta=lam-old;correction=A@delta;constraint=c+correction
    # Natural-map residual in velocity units, avoiding an arbitrary lambda cutoff.
    natural=lam-np.maximum(0.,lam-constraint/np.diag(A))
    residual=float(np.max(abs(natural)*np.diag(A)))
    return dict(impulse=lam,increment=delta,accepted=residual<tolerance,
        projected_residual_m_s=residual,minimum_gap_after_m=float(np.min(gap+h*correction)),
        minimum_velocity_after_m_s=float(np.min(v+correction)),
        optimizer_success=bool(res.success),optimizer_message=str(res.message),iterations=int(res.nit))
