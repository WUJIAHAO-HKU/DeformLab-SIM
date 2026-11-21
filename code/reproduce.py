"""Recompute the paper's figure metrics from the anonymous numerical package."""
from pathlib import Path
import argparse,csv,json,collections
import numpy as np
from transfer_task import score
from core_checks import shared_cap, physical_accounting, refresh_policy, reported_shape
from confirmation import franka_confirmation, refresh_confirmation
from placement_thickness import continuous_placement, thickness_control

DATA=Path(__file__).resolve().parents[1]/'data'
def j(n):return json.loads((DATA/n).read_text(encoding='utf-8-sig'))
def franka():
    cases=j('franka/score-inputs.json'); ref=j('franka/summary.json'); assert len(cases)==18
    rows=[]
    for name,r in cases.items():
        s=r['slot']; rs=[(np.array(k['center']),np.array(k['full_size'])) for k in s['rails']]
        with np.load(DATA/'franka/trajectories'/f'{name}.npz',allow_pickle=False) as f:
            t=f['time_s']; x=f['centroid_m']; z=f['low_m']; hand=f['hand_xyz']; y=f['margin_m']; floor=s['support_top_m']
            lift=(t>=3.7)&(t<=5.0); final=(t>=9.5); col=(t>=5.0)&(t<=9.5)
            relative=x-hand; drift=float(np.linalg.norm(relative[lift]-relative[lift][0],axis=1).max()); carry=float(x[np.argmin(abs(t-5.0)),0]-x[0,0]);
            clear=float((z[lift]-floor).min()); finger=f['finger_positions'][final].min(); h=float((hand[final,2]-f['top_m'][final]).min());
            speed=float(np.quantile(np.linalg.norm(np.diff(x[final],axis=0),axis=1)*960,.95)); jac=float(f['minimum_J'].min()); rail=float(f['rail_penetration_m'][col].max()); margin=float(y[final].min()); height=float(np.abs(z[final]-floor).max())
            criteria=dict(unsupported_pick=clear>.035,retained_in_gripper=drift<.007,carried_to_slot=carry>.040,whole_object_in_opening=margin>=0,deposited_on_floor=height<=.001,released_and_retreated=bool(finger>=.038 and h>=.100),stable_after_withdrawal=speed<.025,valid_elements=jac>0,rail_penetration_below_1mm=rail<=.001)
            a=dict(success=all(criteria.values()),criteria=criteria,final_center_error_m=float(np.median(np.linalg.norm(x[:,:2]-np.array(s['center_xy']),axis=1)[final])),
                final_minimum_margin_m=margin,peak_sampled_rail_penetration_m=rail,minimum_lift_support_clearance_m=clear,
                lift_relative_drift_m=drift,carry_m=carry,minimum_J=jac,settled_speed_p95_m_s=speed,final_floor_height_error_m=height,minimum_hand_to_block_top_m=h)
        if name=='C01-D':
            with np.load(DATA/'franka/raw-example/C01-D.npz',allow_pickle=False) as raw:
                physical,_=score(raw['time_s'],raw['volume_nodes'],raw['hand_poses'],raw['joint_positions'],raw['minimum_J'],np.array(r['block_center_m']),floor,np.array(s['center_xy']),np.array(s['half_opening_m']),rs)
            assert a['criteria']==physical['criteria'] and abs(a['final_center_error_m']-physical['final_center_error_m'])<1e-10
        b=r['task_result']; assert a['criteria']==b['criteria'],(name,a,b)
        for k,v in a.items():
            if k not in ('criteria','success'):assert np.isclose(v,b[k],atol=1e-7 if k=='settled_speed_p95_m_s' else 1e-10,rtol=0),(name,k,v,b[k])
        rows.append(dict(cell=r['case']['id'],method=name.split('-',1)[1],success=bool(a['success']),error_mm=a['final_center_error_m']*1000,margin_mm=a['final_minimum_margin_m']*1000,runtime_s=r['timing']['rollout_s'],supplemental_rows=r['supplemental_row_rounds']))
    out={}
    for m in ('B','D','Dense'):
        g=[x for x in rows if x['method']==m]; b=ref['methods'][m]
        assert len(g)==b['n']==6 and sum(x['success'] for x in g)==b['successes']
        out[m]={'pass':b['successes'],'n':6}
        for k,mapped in [('error_mm','error_mm'),('margin_mm','margin_mm'),('runtime_s','rollout_s')]:
            v=float(np.median([x[k] for x in g]));assert abs(v-b['median_'+mapped])<1e-7
            out[m]['median_'+k]=v
    pairs=[]
    for cell in sorted(set(x['cell'] for x in rows)):
        q={x['method']:x for x in rows if x['cell']==cell};pairs.append(1-q['D']['supplemental_rows']/q['Dense']['supplemental_rows'])
    assert np.allclose(pairs,ref['paired_D_vs_Dense']['supplemental_row_rounds']['savings_fractions'])
    return {'methods':out,'D_supplemental_row_savings':pairs,'rows':rows}

def franka_tolerance_sensitivity():
    cases=j('franka/score-inputs.json')
    names={'B':'Consistent State Mapping','D':'DeformLabSim','Dense':'Full Boundary Contact'}
    tolerances_mm=(0.0,0.05,0.10,0.25,0.50,1.0)
    records=[]
    for key,r in sorted(cases.items()):
        method=names[key.split('-',1)[1]]
        with np.load(DATA/'franka/trajectories'/f'{key}.npz',allow_pickle=False) as f:
            mask=(f['time_s']>=9.5)&(f['time_s']<=10.0)
            margin_mm=float(f['margin_m'][mask].min())*1000
        original=r['task_result']
        assert np.isclose(margin_mm,1000*original['final_minimum_margin_m'],atol=1e-7,rtol=0)
        fixed={k:bool(v) for k,v in original['criteria'].items() if k!='whole_object_in_opening'}
        assert all(fixed.values()),(key,fixed)
        outcomes={str(t):bool(all(fixed.values()) and margin_mm>=-t) for t in tolerances_mm}
        assert outcomes['0.0']==bool(original['success'])
        records.append({'case':r['case']['id'],'method':method,'minimum_final_clearance_mm':margin_mm,'unchanged_other_criteria':fixed,'success_by_tolerance_mm':outcomes})
    assert len(records)==18
    counts={str(t):{m:sum(x['success_by_tolerance_mm'][str(t)] for x in records if x['method']==m) for m in names.values()} for t in tolerances_mm}
    assert list(counts['0.0'].values())==[2,5,6]
    assert list(counts['0.1'].values())==[4,5,6]
    reclassified=[x['case'] for x in records if x['method']==names['B'] and not x['success_by_tolerance_mm']['0.0'] and x['success_by_tolerance_mm']['0.1']]
    assert reclassified==['C03','C06']
    return {'primary_tolerance_mm':0.0,'scope':'Post-hoc rescoring of the same 18 recorded trajectories; only the final whole-object clearance threshold changes. All other criteria and original scores remain fixed.','cases_per_method':6,'successes_by_tolerance_mm':counts,'baseline_reclassified_at_0_10_mm':reclassified,'records':records}

def admissions():
    report=j('admissions/figure3-plot-data.json')
    with np.load(DATA/'admissions/per-node-per-step.npz',allow_pickle=False) as a:
        z=a['selected_rounds']; t=a['step_time_s']; assert z.shape==(7200,125)
        assert int(z.sum())==report['total_selected_row_rounds']
        assert np.allclose(z.reshape(150,48,125).sum(axis=(1,2))/(48*33),report['temporal_curve']['mean_selected_rows_per_round'])
        windows=[]
        for row in report['admission_maps']:
            stop=int(round(row['time_s']*960)); vec=z[stop-48:stop].sum(axis=0); assert int(vec.sum())==row['selected_row_rounds'] and np.count_nonzero(vec)==row['active_nodes']
            windows.append(dict(time_s=row['time_s'],rows=int(vec.sum()),nonzero_nodes=int(np.count_nonzero(vec)),admission_fraction=(vec/(48*33)).tolist()))
    return {'row_rounds':int(z.sum()),'windows':windows}

def gauge():
    a=j('gauge/72-runs.json'); assert len(a)==72
    d=collections.defaultdict(list)
    for r in a:d[(r['observer'],r['condition'],r['arm'])].append(r)
    assert len(d)==12 and all(len(v)==6 for v in d.values())
    assert {m:sum(x['task_success'] for x in a if x['observer']=='depth' and x['arm']==m) for m in ('B','D','dense','frozen_first')}==dict(B=3,D=11,dense=11,frozen_first=9)
    return {','.join(k):{'pass':sum(x['task_success'] for x in v),'n':len(v),'hold_mae_mm':1000*float(np.mean([x['hold_mae_m'] for x in v]))} for k,v in d.items()}

def discovery():
    a=j('discovery/16-runs.json')['points']; assert len(a)==16
    d={m:[r for r in a if r['arm']==m] for m in {r['arm'] for r in a}}
    out={m:{'support_penetration_um': [min(x['peak_um'] for x in rs),max(x['peak_um'] for x in rs)],'median_work_percent_dense':float(np.median([x['row_rounds_percent_dense'] for x in rs]))} for m,rs in d.items()}
    reductions=[100-x['row_rounds_percent_dense'] for x in d['D']]; assert 87.2<min(reductions) and max(reductions)<87.81
    return {'summary':out,'D_reductions_percent':reductions}

def scaling():
    d=j('scaling/frozen-summary.json');rs=[x for x in d['pairs'] if not x['row_diagnostics']]
    assert len(rs)==18 and sum(x['both_timing_valid'] for x in rs)==17
    t=collections.defaultdict(list)
    times={x['case']:x['loop_s'] for x in d['rows'] if not x['row_diagnostics']}
    for r in rs:
        if r['both_timing_valid']:t[(r['condition'],r['resolution'])].append([times[r['D_case']],times[r['dense_case']]])
    r={f'{c},{(n+1)**3} nodes':float(100*(1-np.mean(v,axis=0)[0]/np.mean(v,axis=0)[1])) for (c,n),v in t.items()}
    assert sorted(round(v,2) for k,v in r.items() if '2197 nodes' in k)==[24.81,26.42]
    dh=[1000*x['hold_mae_mm_D_minus_dense'] for x in rs]; assert max(abs(x) for x in dh)<6
    return {'mean_loop_time_reduction_pct':r,'D_minus_dense_hold_errors_um':dh,'physical_pairs':len(rs),'timing_pairs':sum(map(len,t.values()))}

def heldout():
    raw=j('heldout/30-runs.json');a=j('heldout/paired-cost.json');b=j('heldout/bootstrap.json')
    assert len(raw)==30 and len(a['D_vs_Dense_pairs'])==6
    vals=np.array([1-x['D_cost_s']/x['Dense_cost_s'] for x in a['D_vs_Dense_pairs']]); assert np.allclose(vals,b['D_vs_Dense_paired_saving_fraction']['paired_values'])
    import itertools
    # Only six cells: enumerate the full paired case bootstrap distribution as a check.
    drawn=vals[np.array(list(itertools.product(range(6),repeat=6)))].mean(axis=1)
    ci=np.quantile(drawn,(.025,.975)); old=b['D_vs_Dense_paired_saving_fraction']['bootstrap_95_ci']; assert np.allclose(ci,[old[0],old[-1]],atol=.001)
    o={m:sum(r['task_success'] for r in raw if r['method']==m) for m in a['methods']};assert o=={m:r['successes'] for m,r in a['methods'].items()}
    return {'passes_out_of_6':o,'paired_mean_saving_fraction':float(vals.mean()),'paper_bootstrap_95_interval':[old[0],old[-1]],'independent_exhaustive_bootstrap_95_interval':ci.tolist()}

def refresh():
    a=j('refresh/plot-data.json'); raw=j('refresh/development-runs.json'); indep=j('refresh/independent-runs.json')
    assert len(raw)==a['development_runs']==6 and len(indep)==24 and a['independent_runs']==16
    wh=a['selected_trigger']; assert len(wh['eta_m'])==len(wh['gap_m'])==wh['rounds']==25
    p=[x['refresh_fraction'] for x in raw if 'prefinal' in x['method'].lower().replace('-','')]; assert len(p)==2
    assert len(a['independent_D_prefinal_margins_um'])==8
    return {'refresh_fractions_prefinal':p,'independent_pairs_um':a['independent_D_prefinal_margins_um'],'support_penetration_um':a['peak_support_penetrations_um'],'selected_trigger':wh}

def main():
    options={x.__name__:x for x in (admissions,franka,gauge,discovery,scaling,heldout,refresh,shared_cap,physical_accounting,refresh_policy,reported_shape,franka_tolerance_sensitivity,franka_confirmation,refresh_confirmation,continuous_placement,thickness_control)}
    ap=argparse.ArgumentParser();ap.add_argument('--study',choices=list(options));ap.add_argument('--output',type=Path);args=ap.parse_args()
    out={n:f() for n,f in options.items() if not args.study or args.study==n}; blob=json.dumps(out,indent=2,allow_nan=False)+'\n'
    if args.output:args.output.write_text(blob,encoding='utf8')
    else:print(blob,end='')

if __name__=='__main__':main()
