"""Replay actual SDK material/shape readback without replacing it by authored USD."""
from pathlib import Path
import argparse,hashlib,json
from side_push_friction_readback import validate

def audit(folder):
    p=Path(folder);read=lambda n:json.loads((p/n).read_text())
    saved=read('friction-native.json');g=read('geometry.json');rec=read('process-receipt.json')
    actual=validate(saved['raw_sdk'],g['tool_initial_world_center_m'])
    checks=dict(exact_replay=actual==saved,receipt_bound=rec['native_friction_readback']==saved,
                requested_coefficient=rec['case']['config']['material']['friction']==.4)
    return dict(record_valid=all(checks.values()),checks=checks,readback=actual,
                input_sha256={n:hashlib.sha256((p/n).read_bytes()).hexdigest()for n in ['friction-native.json','geometry.json','process-receipt.json']},
                scope='SDK binding/coefficient replay; no independent measurement of the resulting friction impulse')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',required=True);p.add_argument('--out',required=True);a=p.parse_args()
    r=audit(a.run);Path(a.out).write_text(json.dumps(r,indent=2)+'\n');raise SystemExit(0 if r['record_valid']else 1)
