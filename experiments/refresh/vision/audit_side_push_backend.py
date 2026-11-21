"""Actual backend execution coverage, distinct from task success/row activity."""
from pathlib import Path
import argparse,hashlib,json
import numpy as np

def counters_valid(counts,arm,steps,iterations):
    a=np.asarray(counts,dtype=np.uint64);r=iterations+1
    if a.shape!=(steps,24):return False
    k=np.arange(1,steps+1,dtype=np.uint64);expected=k*r
    if not np.array_equal(a[:,0],k) or not np.all(a[:,1:4]==expected[:,None])or np.any(a[:,7]):return False
    s=a[:,8:]
    if arm=='B':return bool(np.all(s==0))
    if arm not in ['D','dense','frozen_first']:return False
    return bool(np.all(s[:,[0,1,7,8]]==expected[:,None]) and np.all(s[:,[2,10,14]]==k[:,None])
        and np.all(s[:,[3,15]]==r)and not np.any(s[:,[4,5,12,13]])and np.all(s[:,6]>0)
        and np.all(s[1:,9]>=s[:-1,9]))

def audit(folder):
    p=Path(folder);rec=json.loads((p/'process-receipt.json').read_text());c=rec['case'];config=c['config'];adapter=rec['adapter']
    a=np.load(p/'trajectory.npz');geo=json.loads((p/'geometry.json').read_text());counts=a['adapter_counts']
    n=round(config['duration_s']*config['solver']['hz']);it=config['solver']['position_iterations'];native=adapter['cp']['native_settings']
    checks=dict(mode=adapter['arm']==c['arm']and adapter['cp']['mode']=='split-velocity',
        all_dispatches=counters_valid(counts,c['arm'],n,it),
        gravity=bool(np.allclose(geo['sdk_initial']['gravity_m_s2'],[0,0,-9.81],atol=1e-6,rtol=0)),
        native_iterations=native['position_iterations']==it and native['velocity_iterations']==1,
        material=bool(np.isclose(native['youngs_modulus'],config['material']['young_pa'],atol=.01,rtol=0)and np.isclose(native['poissons_ratio'],.3,atol=1e-7,rtol=0)and native['material_model']==0),
        damping=all(native[k]==0 for k in ('linear_damping','elasticity_damping','settling_damping','settling_threshold','sleep_threshold')),
        scene=geo['sdk_initial']['solver_is_tgs']and geo['sdk_initial']['external_forces_every_iteration_tgs'],
        surface_finish=(adapter.get('legacy_surface_finish')is True and adapter['surface']['execution_verified']is True)if c['arm'] in ['D','dense','frozen_first']else'surface'not in adapter,
        cleanup=adapter.get('stop_error_count')==0 and adapter['stop_attempted']is True)
    row_count=int(counts[-1,17])if len(counts)and c['arm'] in ['D','dense','frozen_first']else 0
    checks['row_activity_receipt']=adapter['surface_branch_triggered']==(row_count>0)
    return dict(record_valid=bool(all(checks.values())),checks=checks,actual_arm=c['arm'],frames=n,rounds=n*(it+1),
        added_rows=row_count,augmentation_actually_added_rows=row_count>0,
        surface_snapshot_final=counts[-1,8:].tolist()if len(counts)else None,
        scope='Backend activation, exact call coverage and actual material/gravity metadata; row inactivity is an outcome, not task success or universal correctness.',
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        input_sha256={name:hashlib.sha256((p/name).read_bytes()).hexdigest()for name in ('trajectory.npz','process-receipt.json','geometry.json')})
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',required=True);p.add_argument('--out',required=True);a=p.parse_args();r=audit(a.run)
    Path(a.out).write_text(json.dumps(r,indent=2)+'\n');raise SystemExit(0 if r['record_valid']else 1)
