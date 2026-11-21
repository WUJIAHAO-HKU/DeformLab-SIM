"""Adaptive Radau finite-strain/gamma plate reference with planar contact events.

Fixed planar normals and one translating plate. Gamma, elasticity, contact and
drive are coupled within each smooth step. Integrated Radau displacements carry
contact gaps; instantaneous event forces include the same damping. Direct full
versus two-half differences are used without assuming asymptotic order at stiff
transients. No position projection, native acceptance or general robot support.
"""
import numpy as np
from scipy.optimize import brentq
from .radau_nonlinear_active_plate import step
from .damped_plate_contact_events import instantaneous,plastic_impact,acceleration_complementarity


def separating_zero_gap_return(gap,initial_velocity,right,time_tolerance):
    """Factor only an EXACT zero-gap separating start; preserve positive-time gaps."""
    if gap(0.)!=0 or initial_velocity<=0 or right<=0 or gap(right)>0:
        raise ValueError('Zero gap, separating start and closing right bracket required')
    def value(dt):
        return float(initial_velocity) if dt==0 else float(gap(dt)/dt)
    return brentq(value,0.,right,xtol=time_tolerance,rtol=1e-12)


def clock_interval(start,duration,endpoint=None):
    """Keep microscopic event halves distinct; never accept a zero-time advance."""
    start=np.longdouble(start);duration=np.longdouble(duration)
    middle=start+duration/2
    end=start+duration if endpoint is None else np.longdouble(endpoint)
    if not start<middle<end:
        raise RuntimeError('Event interval is below extended clock resolution')
    return start,middle,end


def precise_time(value):
    return np.format_float_scientific(np.longdouble(value),precision=25,unique=False)


def integrate(model,mass,x,v,q,w,bottom,top,table,target,t0,t1,max_step=1/960,min_step=1e-8,
              rtol=2.5e-4,position_atol=1e-9,velocity_atol=1e-6,force_atol=1e-6,
              force_event_tolerance=1e-8,gap_tolerance=1e-12,event_time_tolerance=1e-22,
              max_attempts=20000,on_accept=None,on_event=None,contact_error_control=True,time_breakpoints=(),**physical):
    candidates=(np.asarray(bottom,int),np.asarray(top,int))
    state=(np.array(x,float,copy=True),np.array(v,float,copy=True),float(q),float(w))
    t=np.longdouble(t0);h=float(min(max_step,t1-t));ledger=[];events=[];trials=[]
    target_function=target
    target=lambda time:float(target_function(float(time)))
    breaks=np.asarray(time_breakpoints,float)
    if (breaks.ndim!=1 or not np.isfinite(breaks).all() or np.any(np.diff(breaks)<=0)
            or np.any(breaks<=t0) or np.any(breaks>=t1)):
        raise ValueError('Strictly increasing interior command breakpoints required')
    state_gaps=np.r_[state[0][candidates[0],2]-table,state[2]-state[0][candidates[1],2]]
    initial_normal=np.r_[state[1][candidates[0],2],state[3]-state[1][candidates[1],2]]
    initially_resting=(abs(state_gaps)<=gap_tolerance) & (initial_normal==0)
    split=len(candidates[0])
    active=[candidates[0][initially_resting[:split]],candidates[1][initially_resting[split:]]]
    gap_lookup={id(state[0]):state_gaps}
    work=dict(fixed_contact_solves=0,trial_evaluations=0,trial_cache_hits=0)
    def solve(*args,**kwargs):
        work['fixed_contact_solves']+=1
        return step(*args,record_motion=True,**kwargs)
    if t1<=t0 or min(max_step,min_step,rtol,position_atol,velocity_atol,force_atol)<=0:raise ValueError('Invalid hybrid settings')
    def reactions(s,time):
        _,_,lower,upper=instantaneous(model,mass,*s,*active,target(time),**physical)
        return np.r_[lower,upper]
    def normal_data(s):
        xx,vv,qq,ww=s
        raw=np.r_[xx[candidates[0],2]-table,qq-xx[candidates[1],2]]
        return gap_lookup.get(id(xx),raw),np.r_[vv[candidates[0],2],ww-vv[candidates[1],2]]
    def active_mask():return np.r_[np.isin(candidates[0],active[0]),np.isin(candidates[1],active[1])]
    def gap_increment(result):
        soft=np.asarray(result[4]['integrated_soft_displacement_m'])
        upper=np.asarray(result[4]['integrated_upper_gap_increment_m'])
        return np.r_[soft[candidates[0],2],upper[candidates[1]]]
    def emit(event):
        event['time_precise_s']=precise_time(event['time'])
        event['time']=float(event['time'])
        events.append(event)
        if on_event:on_event(event,state)
    def release(selected,reason):
        old=[ids.copy() for ids in active];n=len(old[0])
        removed=[old[0][selected[:n]],old[1][selected[n:]]]
        active[0]=old[0][~selected[:n]];active[1]=old[1][~selected[n:]]
        emit(dict(type='release',time=t,reason=reason,lower_nodes=removed[0].tolist(),upper_nodes=removed[1].tolist(),
                  position_jump_m=0.,velocity_jump_m_s=0.,kinetic_change_J=0.,material_change_J=0.,impulse_Ns=0.))
    for attempt in range(max_attempts):
        if t>=t1:break
        gap_lookup={id(state[0]):state_gaps}
        if len(events)>1000:raise RuntimeError('Hybrid contact event budget exhausted')
        lam0=reactions(state,t)
        if np.min(lam0,initial=0)<-force_event_tolerance:
            release(lam0<-force_event_tolerance,'inadmissible instantaneous active force');continue
        gaps0,vn0=normal_data(state)
        if gaps0.min(initial=0)<-gap_tolerance:raise RuntimeError('Hybrid state penetrated before trial')
        closed=abs(gaps0)<=gap_tolerance
        if np.any(closed & (gaps0<=0) & (vn0<0)):
            lo=candidates[0][closed[:len(candidates[0])]];up=candidates[1][closed[len(candidates[0]):]]
            new,w1,impact=plastic_impact(mass,*state,lo,up,table,plate_mass=physical.get('plate_mass',.2),gap_tolerance=gap_tolerance)
            state=(state[0],new,state[2],w1)
            _,vn=normal_data(state);keep=closed & (vn==0)
            active=[candidates[0][keep[:len(candidates[0])]],candidates[1][keep[len(candidates[0]):]]]
            emit(dict(type='impact',time=t,lower_nodes=lo.tolist(),upper_nodes=up.tolist(),**impact));continue
        # A zero-gap, zero-speed node can require activation without an impact.
        # Resolve the coupled acceleration inequalities, rather than allowing
        # an inactive resting node to accelerate through the moving plane.
        resting=closed & (vn0==0)
        rest_ids=np.flatnonzero(resting);split=len(candidates[0])
        lo=candidates[0][resting[:split]];up=candidates[1][resting[split:]]
        force,acc=acceleration_complementarity(model,mass,*state,lo,up,target(t),**physical)
        oldmask=active_mask();newmask=np.zeros_like(oldmask)
        keep=(force>force_event_tolerance) | (oldmask[rest_ids] & (abs(acc)<1e-8))
        newmask[rest_ids[keep]]=True
        if not np.array_equal(oldmask,newmask):
            removed=oldmask & ~newmask;added=newmask & ~oldmask
            active=[candidates[0][newmask[:split]],candidates[1][newmask[split:]]]
            for mask,event_type in ((removed,'release'),(added,'activation')):
                if mask.any():emit(dict(type=event_type,time=t,reason='resting acceleration complementarity',
                    lower_nodes=candidates[0][mask[:split]].tolist(),upper_nodes=candidates[1][mask[split:]].tolist(),
                    position_jump_m=0.,velocity_jump_m_s=0.,kinetic_change_J=0.,material_change_J=0.,impulse_Ns=0.))
            continue
        future=breaks[breaks>t]
        boundary=float(future[0]) if len(future) else t1
        h=float(min(h,t1-t,boundary-t));cache={};gap_scores={}
        def trial(dt):
            dt=float(dt)
            if dt==0:return state,None,None,0.
            if dt in cache:
                work['trial_cache_hits']+=1
                return cache[dt]
            work['trial_evaluations']+=1
            tm=t+.5*dt;te=t+dt
            full=solve(model,mass,*state,*active,table,target(t),target(te),dt,**physical)
            half1=solve(model,mass,*state,*active,table,target(t),target(tm),.5*dt,**physical)
            half2=solve(model,mass,*half1[:4],*active,table,target(tm),target(te),.5*dt,**physical)
            # Use the actual h*B*stage-velocity displacement, before rounding
            # it into world coordinates. Trapezoidal endpoint velocities are
            # not the position integrator of Radau and would invent gap drift.
            gap_mid=state_gaps+gap_increment(half1)
            gap_end=gap_mid+gap_increment(half2)
            gap_lookup[id(half1[0])]=gap_mid;gap_lookup[id(half2[0])]=gap_end
            def error(a,b,absolute,scale=None):
                if scale is None:scale=np.maximum(abs(a),abs(b))
                return float(np.max(abs(a-b)/(absolute+rtol*scale),initial=0))
            d1=np.r_[(full[0]-state[0]).ravel(),full[2]-state[2]];d2=np.r_[(half2[0]-state[0]).ravel(),half2[2]-state[2]]
            u1=np.r_[full[1].ravel(),full[3]];u2=np.r_[half2[1].ravel(),half2[3]]
            f=lambda r:np.r_[r['lower_reactions_N'],r['upper_reactions_N']]
            score=max(error(d1,d2,position_atol),error(u1,u2,velocity_atol),error(f(full[4]),.5*(f(half1[4])+f(half2[4])),force_atol))
            # Small opening arcs can be hidden beneath the world-position
            # absolute tolerance. Compare the two *integrated* contact gaps,
            # using the existing geometry tolerance without moving geometry.
            gap_full=state_gaps+gap_increment(full)
            gap_scores[dt]=error(gap_full,gap_end,gap_tolerance)
            if contact_error_control:score=max(score,gap_scores[dt])
            cache[dt]=(half2[:4],half1,half2,score)
            return cache[dt]
        try:s1,half1,half2,score=trial(h)
        except (RuntimeError,ValueError) as exc:
            trials.append(dict(t=float(t),h=h,accepted=False,error=str(exc)))
            if h<=min_step:raise
            h=max(min_step,.5*h);continue
        if score>1:
            trials.append(dict(t=float(t),h=h,accepted=False,error_ratio=score))
            if h<=min_step:raise RuntimeError('Hybrid error control minimum step reached')
            h=max(min_step,h*max(.1,min(.5,.8/score**.25)));continue
        # Sample the interior as well as the endpoint before bracketing the
        # earliest event. Event functions use the same two-half-step trajectory.
        try:
            scan=[0.,h/3,2*h/3,h];scan_states=[state,trial(h/3)[0],trial(2*h/3)[0],s1];event_candidates=[]
            for a,b,sa,sb in zip(scan,scan[1:],scan_states,scan_states[1:]):
                ra=reactions(sa,t+a);rb=reactions(sb,t+b)
                if len(rb) and rb.min()<-force_event_tolerance:
                    if np.min(ra)<=0:
                        root=a
                    else:
                        root=brentq(lambda dt:float(reactions(trial(dt)[0],t+dt).min()),a,b,
                                    xtol=event_time_tolerance,rtol=1e-12)
                    event_candidates.append((root,'release',None))
                ga,_=normal_data(sa);gb,vnb=normal_data(sb);inactive=~active_mask()
                # Root detection uses the actual zero crossing. The acceptance
                # tolerance is not a penetration allowance before impact.
                for index in np.flatnonzero(inactive & (gb<=0) & (vnb<0)):
                    left=a
                    vn_a=normal_data(sa)[1][index]
                    if ga[index]==0 and vn_a>0:
                        root=a+separating_zero_gap_return(
                            lambda offset:normal_data(trial(a+offset)[0])[0][index],
                            vn_a,b-a,event_time_tolerance)
                        event_candidates.append((root,'impact',int(index)))
                        continue
                    # A just-released node starts at zero gap. Find its positive
                    # separation before locating a later descending zero crossing.
                    if ga[index]<=0:
                        found=False
                        for power in range(3,31):
                            fraction=2.**(-power)
                            probe=a+fraction*(b-a)
                            if probe-a<=event_time_tolerance:break
                            if normal_data(trial(probe)[0])[0][index]>0:left=probe;found=True;break
                        if not found:raise RuntimeError('Unresolved immediate recontact; no position projection: '+repr(dict(
                            t=float(t),h=h,index=int(index),ga=float(ga[index]),gb=float(gb[index]),
                            initial_gap=float(gaps0[index]),initial_velocity=float(vn0[index]),active=active_mask().tolist())))
                    root=brentq(lambda dt:float(normal_data(trial(dt)[0])[0][index]),left,b,
                                xtol=event_time_tolerance,rtol=1e-12)
                    event_candidates.append((root,'impact',int(index)))
                # States beyond the first bracket already violate the old contact
                # mode and must not be interpreted as another immediate collision.
                if event_candidates:break
        except (RuntimeError,ValueError) as exc:
            trials.append(dict(t=float(t),h=h,accepted=False,phase='event-search',error=str(exc)))
            if h<=min_step:raise
            h=max(min_step,.5*h);continue
        event=min(event_candidates,key=lambda p:p[0]) if event_candidates else None
        if event is not None:
            dt,kind,event_index=event
            if dt<=event_time_tolerance:
                if kind=='release':
                    selected=lam0<=force_event_tolerance
                    if not selected.any():raise RuntimeError('Zero-time release without vanishing reaction')
                    release(selected,'zero-force event');continue
                raise RuntimeError('Unresolved zero-time collision')
            s1,half1,half2,score=trial(dt)
            if score>1:
                trials.append(dict(t=float(t),h=dt,accepted=False,error_ratio=score,event=kind))
                next_h=max(min_step,.5*dt)
                if next_h>=h or dt<=min_step:
                    _,normal_velocity=normal_data(state)
                    raise RuntimeError('Event refinement cannot reduce step: '+repr(dict(t=float(t),h=h,event_dt=dt,
                        score=score,event=kind,max_active_relative_velocity=float(np.max(abs(normal_velocity[active_mask()]),initial=0.)))))
                h=next_h;continue
        else:dt=h;kind=None;event_index=None
        gmid,_=normal_data(half1[:4]);gend,_=normal_data(s1)
        if min(gmid.min(initial=0),gend.min(initial=0))<-gap_tolerance:
            raise RuntimeError('Event endpoint penetration exceeds tolerance')
        if not (half1[4]['unilateral_admissible'] and half2[4]['unilateral_admissible']):
            trials.append(dict(t=float(t),h=dt,accepted=False,reason='negative stage reaction'))
            if dt<=min_step:raise RuntimeError('Stage force admissibility failed at minimum step')
            h=max(min_step,.5*dt);continue
        endpoint=boundary if kind is None and dt==float(boundary-t) else None
        start_time,middle_time,end_time=clock_interval(t,dt,endpoint)
        trials.append(dict(t=float(t),h=dt,accepted=True,error_ratio=score,event=kind,
                           contact_gap_error_ratio=gap_scores[dt]))
        for start,end,result in ((start_time,middle_time,half1),(middle_time,end_time,half2)):
            tracked,_=normal_data(result[:4]);xx,_,qq,_=result[:4]
            geometric=np.r_[xx[candidates[0],2]-table,qq-xx[candidates[1],2]]
            discrepancy=float(np.max(abs(tracked-geometric),initial=0))
            if discrepancy>gap_tolerance:raise RuntimeError('Contact increment and world positions disagree')
            row=dict(result[4],t0=float(start),t1=float(end),h=.5*dt,
                t0_precise_s=precise_time(start),t1_precise_s=precise_time(end),
                clock_quantization_s=float(end-start-np.longdouble(.5*dt)),
                lower_nodes=active[0].tolist(),upper_nodes=active[1].tolist(),
                contact_gaps_m=tracked.tolist(),gap_reconstruction_error_m=discrepancy)
            ledger.append(row)
            if on_accept:on_accept(float(start),float(end),result[:4],row)
        state_gaps=normal_data(s1)[0].copy();state=s1;t=end_time
        if kind=='release':
            now=reactions(state,t);selected=now<=force_event_tolerance
            if not selected.any():raise RuntimeError('Release root missed force threshold')
            release(selected,'located zero-force event')
        elif kind=='impact':
            # The located event defines this contact coordinate as zero.
            # Reinitialize only that scalar root residual; world positions,
            # material state and velocities are unchanged. Retain the exact
            # correction and check it against the original geometry gate.
            root_residual=float(state_gaps[event_index])
            if abs(root_residual)>gap_tolerance:raise RuntimeError('Located contact root exceeds geometry tolerance')
            state_gaps[event_index]=0.;gap_lookup[id(state[0])]=state_gaps
            gaps,vn=normal_data(state);closed=abs(gaps)<=gap_tolerance
            lo=candidates[0][closed[:len(candidates[0])]];up=candidates[1][closed[len(candidates[0]):]]
            new,w1,impact=plastic_impact(mass,*state,lo,up,table,plate_mass=physical.get('plate_mass',.2),gap_tolerance=gap_tolerance)
            state=(state[0],new,state[2],w1);_,after=normal_data(state);keep=closed & (after==0)
            active=[candidates[0][keep[:len(candidates[0])]],candidates[1][keep[len(candidates[0]):]]]
            raw=np.r_[state[0][candidates[0],2]-table,state[2]-state[0][candidates[1],2]]
            reconstruction=float(np.max(abs(state_gaps-raw),initial=0.))
            if reconstruction>gap_tolerance:raise RuntimeError('Event-coordinate reinitialization disagrees with geometry')
            emit(dict(type='impact',time=t,lower_nodes=lo.tolist(),upper_nodes=up.tolist(),
                located_candidate_index=event_index,root_gap_residual_m=root_residual,
                contact_coordinate_reinitialization_m=-root_residual,
                post_event_gap_reconstruction_error_m=reconstruction,**impact))
        h=min(max_step,max(min_step,dt*min(2.,max(.5,.85/max(score,1e-12)**.25))))
    if t<t1:raise RuntimeError('Hybrid integration attempt budget exhausted')
    return (*state,dict(t0=t0,t1=t1,ledger=ledger,events=events,trials=trials,
        accepted_substeps=len(ledger),rejected_trials=sum(not r['accepted'] for r in trials),
        contact_error_control=contact_error_control,work_counters=work,
        error_estimator='unscaled full-versus-two-half difference; no assumed asymptotic Richardson factor',
        time_breakpoints=breaks.tolist(),
        clock_representation='extended internal clock; precise timestamps and actual durations accompany float timestamps',
        final_lower_nodes=active[0].tolist(),final_upper_nodes=active[1].tolist()))
