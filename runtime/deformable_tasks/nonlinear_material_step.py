"""Small independent finite-strain midpoint/discrete-work dynamical reference.

Analytic matrix-free Newton or diagnostic finite differences. No contact, damping,
articulation or native/GPU backend support is implied.
"""
import numpy as np
from .corotated_discrete_work import discrete_material_gradient
from .corotated_discrete_tangent import DiscreteMaterialTangent


def step(model,mass,position,velocity,h,tolerance_N=1e-8,max_iterations=30,linear_solver='analytic-gmres'):
    x=np.asarray(position,float);v=np.asarray(velocity,float);m=np.asarray(mass,float)[:,None]
    if m.shape!=(len(x),1) or not np.all(m>0) or not np.isfinite(m).all() or h<=0 or not np.isfinite(h):raise ValueError('Invalid step parameters')
    if linear_solver not in ('analytic-gmres','finite-difference'):raise ValueError('Unknown linear solver')
    model.kinematics(x)
    def evaluate(delta):
        g,info=discrete_material_gradient(model,x,x+delta)
        return (2*m/h**2*(delta-h*v)+g).ravel(),info
    delta=np.zeros_like(x);history=[];linear_iterations=[]
    for iteration in range(max_iterations+1):
        r,info=evaluate(delta);res=float(abs(r).max());history.append(res)
        if res<tolerance_N:break
        if iteration==max_iterations:raise RuntimeError('Nonlinear material step failed to converge: '+repr(history))
        if linear_solver=='finite-difference':
            jac=np.empty((x.size,x.size));eps=1e-7
            for j in range(x.size):
                perturb=np.zeros_like(x);perturb.flat[j]=eps
                rp,_=evaluate(delta+perturb);rm,_=evaluate(delta-perturb);jac[:,j]=(rp-rm)/(2*eps)
            direction=np.linalg.solve(jac,-r).reshape(x.shape)
            linear_iterations.append(0)
        else:
            from scipy.sparse.linalg import LinearOperator,gmres
            tangent=DiscreteMaterialTangent(model,x,x+delta)
            def product(flat):
                d=flat.reshape(x.shape)
                return (2*m/h**2*d+tangent.product(d)).ravel()
            operator=LinearOperator((x.size,x.size),matvec=product,dtype=float)
            preconditioner=LinearOperator(operator.shape,matvec=lambda d:(d.reshape(x.shape)*h**2/(2*m)).ravel(),dtype=float)
            iterations=[]
            direction,status=gmres(operator,-r,M=preconditioner,rtol=1e-10,atol=1e-12,
                restart=min(x.size,100),maxiter=100,callback=iterations.append,callback_type='pr_norm')
            linear_iterations.append(len(iterations))
            if status!=0 or np.linalg.norm(product(direction)+r)>max(1e-11,1e-9*np.linalg.norm(r)):
                raise RuntimeError('Analytic material GMRES failed: '+str(status))
            direction=direction.reshape(x.shape)
        alpha=1.
        for _ in range(30):
            trial=delta+alpha*direction
            try:rt,_=evaluate(trial)
            except ValueError:alpha*=.5;continue
            if np.linalg.norm(rt)<np.linalg.norm(r):delta=trial;break
            alpha*=.5
        else:raise RuntimeError('Positive-volume residual line search failed')
    y=x+delta;new=2*delta/h-v
    U0,_=model.energy_gradient(x);U1,_=model.energy_gradient(y)
    energy=.5*float(np.sum(m*(new**2-v**2)))+U1-U0
    P=np.sum(m*(new-v),axis=0);L=np.sum(np.cross(y,m*new)-np.cross(x,m*v),axis=0)
    info.update(newton_iterations=iteration,force_residual_N=res,residual_history_N=history,
        linear_solver=linear_solver,linear_iterations=linear_iterations,
        total_energy_change_J=energy,linear_momentum_change_Ns=float(np.linalg.norm(P)),
        angular_momentum_change_Nms=float(np.linalg.norm(L)))
    return y,new,info
