"""Recompute actual image feedback and causal command inputs; offline, not a rollout."""
from pathlib import Path
import argparse,hashlib,json,sys
import numpy as np
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'work/paper-generalization-v1/vision/v5'))
sys.path.insert(0,str(ROOT/'work/paper-mechanism-v1/vision'))
from condition_camera import validate_actual_camera
from side_push_observers import observe_color,observe_depth
from side_push_geometry import leading_feature
from side_push_control import InitialPositionCalibration,PositionController
from side_push_config import schedule

def close(a,b):
    if isinstance(a,dict):return isinstance(b,dict) and a.keys()==b.keys() and all(close(v,b[k])for k,v in a.items())
    if isinstance(a,(list,tuple)):return isinstance(b,(list,tuple))and len(a)==len(b)and all(close(x,y)for x,y in zip(a,b))
    if isinstance(a,(float,np.floating)):return isinstance(b,(int,float)) and bool(np.isclose(a,b,atol=1e-12,rtol=0))
    return a==b

def audit(folder):
    folder=Path(folder);rec=json.loads((folder/'process-receipt.json').read_text());case=rec['case']
    geo=json.loads((folder/'geometry.json').read_text());feedback=json.loads((folder/'feedback.json').read_text())
    cal=validate_actual_camera(case['condition']['camera'],geo['camera_usd_row_matrix'],geo['camera_lens_readback'])
    feature=json.loads((folder/'oracle-feature.json').read_text());trajectory=np.load(folder/'trajectory.npz')
    scale=InitialPositionCalibration(.02+case['initial_x_offset_m']);controller=PositionController();rows=[];hashes={}
    gates=dict(actual_camera_readback=True,raw_replay=True,calibration_replay=True,action_replay=True,software_clock=True)
    for i,(packet,action)in enumerate(zip(feedback['packets'],feedback['actions'])):
        step=packet['step'];t=packet['time_s'];path=folder/'rgbd'/f'frame-{step:05d}.npz';image=np.load(path)
        hashes[str(path.name)]=hashlib.sha256(path.read_bytes()).hexdigest();kw=dict(sample_time_s=t,sequence=i)
        if case['observer']=='color':raw=observe_color(image['rgb'],image['depth'],cal,**kw)
        elif case['observer']=='depth':raw=observe_depth(image['depth'],cal,**kw)
        elif case['observer']=='truth':raw=leading_feature(trajectory['positions'][step],feature,**kw)
        else:raise ValueError('Unknown active feedback source')
        ob=scale.apply(raw);s=schedule(t,case['initial_x_offset_m']);expected=controller.update(t,s['target_position_m'],ob,active=s['feedback_active'])
        gates['raw_replay'] &= close(raw,packet['raw_observation'])
        gates['calibration_replay'] &= close(ob,packet['observation']) and close(scale.state(),packet['calibration_state'])
        gates['action_replay'] &= close(expected,action)
        gates['software_clock'] &= abs(step/case['config']['solver']['hz']-t)<1e-10 and abs(t-i*.1)<1e-10 and abs(packet['physics_time_before']-packet['physics_time_after'])<1e-10
        truth=leading_feature(trajectory['positions'][step],feature,**kw)
        rows.append(dict(step=step,time_s=t,valid=ob['valid'],raw_x_m=raw['position_m'],calibrated_x_m=ob['position_m'],
            actual_material_feature_x_m=truth['position_m'],raw_feature_bias_m=raw['position_m']-truth['position_m']if raw['valid']and truth['valid']else None,
            calibrated_feature_bias_m=ob['position_m']-truth['position_m']if ob['valid']and truth['valid']else None,action=expected))
    gates['feedback_count']=len(rows)==round(case['config']['duration_s']*10)+1 and len(feedback['packets'])==len(feedback['actions'])
    return dict(replay_integrity_passed=bool(all(gates.values())),checks=gates,rows=rows,
        image_sha256=hashes,source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        initial_observation_valid=rows[0]['valid']if rows else False,invalid_observations=sum(not x['valid']for x in rows),
        geometry_ray_identity='Not certified by algorithm replay; separate independent actual triangle/fixture ray audit required',
        scope='Same saved inputs and actual active controller replay; not a counterfactual closed-loop rollout. Material-feature truth is used solely for retrospective scoring in image modes.')
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',required=True);p.add_argument('--out',required=True);a=p.parse_args()
    r=audit(a.run);Path(a.out).write_text(json.dumps(r,indent=2,allow_nan=False)+'\n');raise SystemExit(0 if r['replay_integrity_passed']else 1)
