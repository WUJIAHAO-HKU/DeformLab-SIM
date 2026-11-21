"""CPU contracts for new inputs and unchanged image/controller branches."""
from pathlib import Path
import hashlib,json,sys
import numpy as np
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[2]
sys.path.insert(0,str(ROOT/'work/paper-generalization-v1/vision/v5'))
sys.path.insert(0,str(ROOT/'work/paper-mechanism-v1/vision'))
from heldout_protocol import cases,CONDITIONS,ARMS
from side_push_config import target_pose,schedule
from side_push_observers import observe_depth
from side_push_control import PositionController
from condition_camera import validate_actual_camera
from audit_side_push_backend import counters_valid

def test():
    checks={};old=ROOT/'work/paper-method-generalization-v1/vision'
    for n in ['side_push_control.py','side_push_observers.py','side_push_config.py','side_push_geometry.py']:
        # The inherited config has CRLF storage; only normalize newline bytes.
        assert (HERE/n).read_bytes().replace(b'\r\n',b'\n')==(old/n).read_bytes().replace(b'\r\n',b'\n')
    assert (HERE/'side_push_adapter.py').read_text()==(old/'side_push_adapter.py').read_text().replace('from robot_surface_live_bridge import RobotSurfaceLive','from timing_surface_bridge import RobotSurfaceLive')
    checks['controller_observers_geometry_identical_adapter_only_traceoff_import']=True
    allocation=dict(status='fixed_work_functional_only',selected={arm:dict(iterations=it,component_build='dummy/'+arm) for arm,it in zip(ARMS,[48,32,24])})
    cc=cases('confirmation',allocation);assert len(cc)==36 and len({c['name'] for c in cc})==36
    keys={}
    for c in cc:
        key=(c['condition']['name'],c['observer'],c['random_initialization_seed'])
        keys.setdefault(key,[]).append(c)
    assert len(keys)==12
    for group in keys.values():
        assert {c['arm'] for c in group}==set(ARMS)
        assert len({c['initial_x_offset_m'] for c in group})==1
        assert all(c['condition']==group[0]['condition'] for c in group)
        assert all(c['config']['material']==group[0]['config']['material'] for c in group)
    checks['36_paired_cells_no_backend_physical_changes']=True
    for c in cc:
        offset=c['initial_x_offset_m'];assert abs(offset)<=.0025
        for t in np.linspace(0,3.2,33):
            for correction in [-.004,0,.004]:target_pose(float(t),correction,offset)
        assert abs(schedule(3.2,offset)['target_position_m']-.06)<1e-14
    checks['all_new_random_starts_preserve_command_and_target_contract']=True
    for condition in CONDITIONS:
        c=next(c for c in cc if c['condition']['name']==condition['name']);cam=c['condition']['camera']
        cal=validate_actual_camera(cam,np.asarray(cam['expected_usd_row_matrix']),cam['lens'])
        f=cal.fx;distance=condition['camera_distance_m']-.02*condition['geometry_y_scale']
        yy,xx=np.indices((480,640));x=.02+(xx+.5-320)*distance/f;z=.14-(yy+.5-240)*distance/f
        values=[]
        for i,shift in enumerate([0,.02,.04]):
            mask=(x>=-.02+shift)&(x<=.02+shift)&(z>=.12)&(z<=.16)
            depth=np.where(mask,distance,1.).astype(np.float32)
            ob=observe_depth(depth,cal,sample_time_s=.1*i,sequence=i)
            assert ob['valid'],(condition,ob)
            values.append(ob['position_m'])
        assert np.max(abs(np.asarray(values)-values[0]-[0,.02,.04]))<.0005
    checks['synthetic_heldout_camera_fronts_valid_with_unchanged_depth_observer']=True
    for arm in ARMS:
        for it in [16,24,32,48]:
            n=3;r=it+1;k=np.arange(1,n+1,dtype=np.uint64);a=np.zeros((n,24),dtype=np.uint64)
            a[:,0]=k;a[:,1:4]=(k*r)[:,None]
            if arm!='B':
                s=a[:,8:];s[:,[0,1,7,8]]=(k*r)[:,None];s[:,[2,10,14]]=k[:,None];s[:,[3,15]]=r;s[:,6]=125;s[:,9]=k
            assert counters_valid(a,arm,n,it)
            corrupt=a.copy();corrupt[-1,1]+=1;assert not counters_valid(corrupt,arm,n,it)
    checks['dynamic_budget_dispatch_counts_and_dense_branch_mutation_rejection']=True
    assert len(cases('development'))==6
    assert not ({c['name'] for c in cases('development')} & {c['name'] for c in cc})
    try:cases('confirmation')
    except ValueError:pass
    else:raise AssertionError('Unallocated formal accepted')
    checks['development_formal_isolation_and_allocation_required']=True
    return dict(passed=True,checks=checks,scope='CPU input, camera, controller-source and counter-schema contracts; no native geometry cook, rendering, task result or budget certification')
if __name__=='__main__':
    r=test();print(json.dumps(r,indent=2))
    if len(sys.argv)>1:Path(sys.argv[1]).write_text(json.dumps(r,indent=2)+'\n',encoding='utf-8')
