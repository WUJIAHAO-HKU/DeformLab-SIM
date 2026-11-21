"""Independent v2 scorer: actual SDK Bench bounds; all v1 thresholds unchanged."""
from pathlib import Path
import argparse,hashlib,json
import numpy as np
from side_push_geometry import freeze_front,leading_feature,point_box_clearance

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def audit(folder):
    p=Path(folder);rec=json.loads((p/'process-receipt.json').read_text());a=np.load(p/'trajectory.npz')
    case=rec['case'];hz=case['config']['solver']['hz'];n=round(hz*case['config']['duration_s'])
    geometry=json.loads((p/'geometry.json').read_text())['oracle_geometry']
    bench_center=np.asarray(geometry['bench_position_m'],float);bench_half=np.asarray(geometry['bench_half_extents_m'],float)
    assert np.allclose(np.abs(np.asarray(geometry['bench_quaternion_wxyz'], dtype=float)),[1,0,0,0],atol=1e-12,rtol=0)
    tool_half=np.asarray(geometry['tool_half_extents_m'],float)
    x=a['positions'].astype(float);v=a['velocities'].astype(float);t=a['time_s'];rest=a['rest'];tet=a['elements'].astype(int)
    feature=freeze_front(rest,tet);saved=json.loads((p/'oracle-feature.json').read_text());assert feature==saved
    gates={};gates['completed_receipt']=rec['status']=='completed_pending_exit' and not rec.get('errors')
    gates['frame_count']=len(x)==n+1 and len(t)==n+1
    gates['clock']=gates['frame_count'] and bool(np.allclose(t,np.arange(n+1)/hz,atol=1e-10,rtol=0))
    gates['native_clock']=bool(np.allclose(a['native_time_s'],t,atol=1e-6,rtol=0))
    gates['finite']=bool(np.isfinite(x).all() and np.isfinite(v).all())
    gates['no_rollout_reset']=rec['node_resets_during_rollout']==0
    gates['native_cleanup']=rec.get('adapter_cleanup',{}).get('stop_error_count')==0
    count=a['adapter_counts'];expected=np.arange(1,n+1)*(case['config']['solver']['position_iterations']+1)
    gates['cp_dispatch_count']=count.shape==(n,24) and bool(np.all(count[:,1:4]==expected[:,None]))
    gates['zero_attachments']=rec.get('attachments')==[]
    gates['axis_aligned_tool']=bool(np.allclose(abs(a['tool_quaternions']),[1,0,0,0],atol=2e-6,rtol=0))
    ds=rest[tet[:,1:]]-rest[tet[:,0,None]];rest_det=np.linalg.det(ds)
    j=np.empty(len(x));penetration=np.empty(len(x));clearance=np.empty(len(x));measurements=[]
    for i,points in enumerate(x):
        j[i]=np.min(np.linalg.det(points[tet[:,1:]]-points[tet[:,0,None]])/rest_det)
        q=abs(points-a['tool_positions'][i])-tool_half
        penetration[i]=max(0.,-float(np.max(q,axis=1).min()))
        clearance[i]=point_box_clearance(points,a['tool_positions'][i],tool_half)
        measurements.append(leading_feature(points,feature,sample_time_s=t[i],sequence=i))
    values=np.asarray([r['position_m']if r['valid']else np.nan for r in measurements]);valid=np.isfinite(values)
    hold=(t>=2.1-1e-9)&(t<=2.4+1e-9);free=(t>=2.8-1e-9)&(t<=3.2+1e-9)
    def window(mask):
        full=bool(np.all(valid[mask]));part=values[mask&valid]
        return dict(expected_samples=int(mask.sum()),valid_samples=int((mask&valid).sum()),all_feature_samples_valid=full,
            mae_m=float(np.mean(abs(part-.06)))if full and len(part)else None,
            partial_valid_mae_m=float(np.mean(abs(part-.06)))if len(part)else None,
            primary_eligible=full,invalid_samples_retained=int((mask&~valid).sum()))
    footprint=np.all(abs(x[:,:,:2]-bench_center[:2])<bench_half[:2],axis=2)
    support=np.where(footprint,np.maximum(bench_center[2]+bench_half[2]-x[:,:,2],0),0.).max(axis=1)
    support_available=np.any(footprint,axis=1)
    margin=np.min(bench_half[:2]-abs(x[:,:,:2]-bench_center[:2]),axis=(1,2))
    feedback=json.loads((p/'feedback.json').read_text());packets=feedback['packets'];actions=feedback['actions']
    gates['feedback_count']=len(packets)==round(case['config']['duration_s']*10)+1==len(actions)
    gates['render_does_not_step']=all(abs(k['physics_time_before']-k['physics_time_after'])<1e-10 for k in packets)
    gates['control_image_branches']=case['observer']in ['color','depth','truth']
    # Full source-bound command replay, using only the preceding observed action.
    from side_push_config import target_pose
    command_ok=True;correction=0.;cursor=0
    for step in range(n+1):
        while cursor<len(packets) and packets[cursor]['step']<step:
            correction=actions[cursor]['correction_m'];cursor+=1
        expected_command=target_pose(step/hz,correction,case['initial_x_offset_m'])
        command_ok &= bool(np.allclose(a['commanded_tool_positions'][step],expected_command,atol=1e-8,rtol=0))
    gates['strictly_previous_feedback_command']=command_ok
    result=dict(scope='Development task unless explicitly promoted by a later frozen protocol; no universal pushing/release claim',
        execution_checks=gates,execution_valid=all(gates.values()),case=case,
        primary_hold=window(hold),post_withdrawal=window(free),
        final_error_m=float(abs(values[-1]-.06))if valid[-1]else None,
        post_withdrawal_drift_m=float(values[-1]-values[np.flatnonzero(free)[0]])if valid[free].all()else None,
        hold_success=bool(valid[hold].all() and np.mean(abs(values[hold]-.06))<=.002),
        actual_release_clearance_m=float(clearance[free].min()),released=bool(np.all(clearance[free]>.001)),
        post_withdrawal_nodal_rms_speed_m_s=float(np.sqrt(np.mean(v[free]**2))),
        all_frame_min_tet_J=float(j.min()),all_frame_min_support_domain_margin_m=float(margin.min()),
        support_domain_valid=bool(np.all(margin>0)),full_support_peak_m=float(support.max())if support_available.all()else None,
        hold_support_peak_m=float(support[hold].max())if support_available[hold].all()else None,
        support_metric_scope='Only nodes strictly inside actual fixed Bench footprint; any out-of-domain node is independently reported as a domain failure',
        support_unavailable_frames=int((~support_available).sum()),full_side_tool_node_penetration_peak_m=float(penetration.max()),
        initial_observation_valid=packets[0]['observation']['valid'],invalid_observations=sum(not q['observation']['valid']for q in packets),
        source_sha256=sha(__file__),input_sha256={name:sha(p/name)for name in ('trajectory.npz','process-receipt.json','feedback.json','oracle-feature.json','geometry.json')},
        limitations=['Nodal OBB penetration is not continuous triangle collision certification',
            'Material front can lose visibility; image/truth measurement differences require independent ray audit',
            'Execution validity does not require task/settling success; all failures are retained'])
    np.savez_compressed(p/'side-push-score-series.npz',time_s=t,leading_x_m=values,feature_valid=valid,minimum_tet_J=j,
        support_penetration_m=support,support_metric_available=support_available,support_domain_margin_m=margin,side_tool_penetration_m=penetration,pusher_clearance_m=clearance)
    return result
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',required=True);p.add_argument('--out',required=True);args=p.parse_args()
    result=audit(args.run);Path(args.out).write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    raise SystemExit(0 if result['execution_valid']else 1)
