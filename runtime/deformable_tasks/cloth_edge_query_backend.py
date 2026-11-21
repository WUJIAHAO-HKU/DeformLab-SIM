"""Strict artifact and execution proof for the explicit corrected EE query."""
import json,hashlib
from pathlib import Path
from .cloth_stage_backend import sha
BACKEND='cloth-split-envelope-refresh16-edge-query-contact'
def artifacts(root):
 from .cloth_split_backend import artifacts as base_artifacts
 root=Path(root);files=base_artifacts(root,'cloth-split-envelope-refresh16-contact');build=root/'results/cloth-edge-query-backend-build-v1';m=json.loads((build/'build.json').read_text())
 files.update(library=build/'libcloth_split_backend.so',manifest=build/'build.json',cloth_split_ptx=build/'contact.ptx')
 if m['status']!='compiled' or not m.get('explicit_edge_query_start_required') or sha(files['library'])!=m['library_sha256'] or sha(files['cloth_split_ptx'])!=m['ptx_sha256']:raise ValueError('Verified edge-query build required')
 if hashlib.sha256(files['cloth_split_ptx'].read_bytes()[:m['base_contact_ptx_prefix_bytes']]).hexdigest()!=m['base_contact_ptx_sha256'] or sha(m['base_contact_ptx'])!=m['base_contact_ptx_sha256']:raise ValueError('Underlying contact kernels changed')
 pins=dict(m['input_sha256'])
 for key,name in [('cloth_split_selftest','cloth-edge-query-base-binding-v1'),('selftest','cloth-edge-query-dispatch-v1'),('cache_selftest','cloth-edge-query-cache-v1'),('edge_query_selftest','cloth-edge-query-binding-v1')]:
  f=root/'results'/name/'summary.json';r=json.loads(f.read_text())
  if not r['passed']:raise ValueError('Edge-query binding prerequisite failed')
  files[key]=f;pins.update(r['input_sha256'])
 owned=root/'results/cloth-edge-query-fix-owned-v1';r=json.loads((owned/'summary.json').read_text());protocol=owned/'protocol.json'
 if not r['passed'] or r['trials']!=152 or sha(protocol)!=r['protocol_sha256']:raise ValueError('Independent query correction checks required')
 files.update(edge_query_owned=owned/'summary.json',edge_query_protocol=protocol);pins.update(json.loads(protocol.read_text())['input_sha256'])
 if any(sha(p)!=h for p,h in pins.items()):raise ValueError('Edge-query source/test input changed')
 return files

def evidence(values,status,counts):
 v=list(map(int,values));expected=counts.get('cloth_queryClothClothContactEECountLaunch',0)
 valid=status==0 and len(v)==6 and expected>0 and v[0]==v[1]==expected and v[2]>0 and v[3]>0 and v[4]==v[5]==0
 return dict(host_calls=v[0],gpu_launches=v[1],enabled_rows=v[2],active_rows=v[3],rejected_rows=v[4],read_errors=v[5],exit_code=status,execution_verified=bool(valid),physical_improvement_accepted=False,performance_accepted=False,platform_acceptance=False)
