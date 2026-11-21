"""Error-controlled fixed-contact finite-strain reference integration.

Accepts two half steps only after motion AND mean reaction step-doubling checks.
Does not extrapolate states, clip forces, or release contacts. Persistent negative
reaction at minimum step is an explicit failure requiring an event/contact solver.
"""
import numpy as np
from .nonlinear_active_plate import step


def integrate(model,mass,x,v,q,w,bottom,top,table,target,t0,t1,
              max_step=1/960,min_step=1e-8,motion_rtol=1e-3,force_rtol=1e-3,
              position_atol=1e-9,velocity_atol=1e-6,force_atol=1e-6,
              max_attempts=10000,on_accept=None,**physical):
    tolerances=[max_step,min_step,motion_rtol,force_rtol,position_atol,velocity_atol,force_atol]
    if not np.isfinite(tolerances).all() or min(tolerances)<=0 or min_step>max_step or not np.isfinite([t0,t1]).all() or t1<=t0:
        raise ValueError('Invalid adaptive integration parameters')
    current=(np.array(x,float,copy=True),np.array(v,float,copy=True),float(q),float(w))
    t=float(t0);h=min(max_step,t1-t);accepted=[];trials=[]
    def solve(state,a,b):
        return step(model,mass,*state,bottom,top,table,target(a),target(b),b-a,**physical)
    def forces(record):return np.r_[record['lower_reactions_N'],record['upper_reactions_N']]
    def scaled_error(diff,scale,absolute,relative):
        return float(np.max(abs(diff)/(absolute+relative*scale)))
    for attempt in range(max_attempts):
        if t>=t1:break
        h=min(h,t1-t);end=t+h;mid=t+.5*h
        record=dict(t0=t,t1=end,h=h,accepted=False)
        try:
            full=solve(current,t,end);half1=solve(current,t,mid);half2=solve(half1[:4],mid,end)
            # Error estimate for the accepted two-half-step solution; no
            # Richardson extrapolation, which would spoil work/constraints.
            dx_full=np.r_[(full[0]-current[0]).ravel(),full[2]-current[2]]
            dx_fine=np.r_[(half2[0]-current[0]).ravel(),half2[2]-current[2]]
            v_full=np.r_[full[1].ravel(),full[3]];v_fine=np.r_[half2[1].ravel(),half2[3]]
            f_full=forces(full[4]);f_fine=.5*(forces(half1[4])+forces(half2[4]))
            position_error=scaled_error((dx_fine-dx_full)/3,np.maximum(abs(dx_fine),abs(dx_full)),position_atol,motion_rtol)
            velocity_error=scaled_error((v_fine-v_full)/3,np.maximum(abs(v_fine),abs(v_full)),velocity_atol,motion_rtol)
            reaction_error=scaled_error((f_fine-f_full)/3,np.maximum(abs(f_fine),abs(f_full)),force_atol,force_rtol)
            score=max(position_error,velocity_error,reaction_error)
            admissible=half1[4]['unilateral_admissible'] and half2[4]['unilateral_admissible']
            record.update(position_error=position_error,velocity_error=velocity_error,reaction_error=reaction_error,
                error_ratio=score,unilateral_admissible=admissible,
                min_trial_reaction_N=min(full[4]['min_reaction_N'],half1[4]['min_reaction_N'],half2[4]['min_reaction_N']))
        except (RuntimeError,ValueError) as exc:
            score=float('inf');admissible=False;record['solver_error']=str(exc)
        if score<=1 and admissible:
            record['accepted']=True
            for a,b,result in ((t,mid,half1),(mid,end,half2)):
                ledger=dict(result[4],t0=a,t1=b,h=b-a)
                accepted.append(ledger)
                if on_accept is not None:on_accept(a,b,result[:4],ledger)
            current=half2[:4];t=end
            h=min(max_step,h*min(2.,max(.5,.85/max(score,1e-12)**.5)))
        else:
            factor=.5 if not admissible or not np.isfinite(score) else max(.1,min(.5,.8/score**.5))
            if h<=min_step*(1+1e-12):
                raise RuntimeError('Adaptive fixed-contact minimum step reached: '+repr(record))
            h=max(min_step,h*factor)
        trials.append(record)
    if t<t1:raise RuntimeError('Adaptive fixed-contact attempt budget exhausted')
    result=dict(t0=t0,t1=t1,accepted_substeps=len(accepted),rejected_trials=sum(not r['accepted'] for r in trials),
        min_step_s=min(r['h'] for r in accepted),max_step_s=max(r['h'] for r in accepted),
        max_force_residual_N=max(r['force_residual_N'] for r in accepted),
        min_reaction_N=min(r['min_reaction_N'] for r in accepted),
        total_work_residual_J=sum(r['full_work_residual_J'] for r in accepted),ledger=accepted,trials=trials)
    return (*current,result)
