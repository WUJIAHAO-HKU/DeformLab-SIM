"""Finite-feature release at the actual cloth contact envelope, with counted execution."""
import json,math,struct
from pathlib import Path
from .cloth_stage_backend import sha
BACKEND='cloth-split-envelope-refresh16-edge-query-history-surface-release-contact'
def artifacts(root):
 from .cloth_failure_ring_backend import artifacts as base
 root=Path(root);files=base(root);build=root/'results/cloth-surface-release-backend-build-v1';m=json.loads((build/'build.json').read_text())
 if m.get('status')!='compiled' or m.get('surface_envelope_release') is not True or m.get('release_derived_from_rest_offset') is not True:raise ValueError('Counted rest-derived surface release required')
 files.update(library=build/'libcloth_split_backend.so',manifest=build/'build.json',cloth_split_ptx=build/'contact.ptx')
 if sha(files['library'])!=m['library_sha256'] or sha(files['cloth_split_ptx'])!=m['ptx_sha256']:raise ValueError('Surface release build changed')
 pins=dict(m['input_sha256'])
 for key,folder,count in [('cloth_split_selftest','cloth-surface-release-binding-v2',312),('history_selftest','cloth-surface-release-binding-v2',312),('selftest','cloth-surface-release-dispatch-v1',10),('cache_selftest','cloth-surface-release-cache-v1',12),('history_physics_selftest','cloth-surface-release-physics-v1',640),('surface_release_selftest','cloth-surface-release-probe-v1',396)]:
  p=root/'results'/folder/'summary.json';r=json.loads(p.read_text())
  if r.get('passed') is not True or len(r.get('checks',{}))!=count or not all(r['checks'].values()):raise ValueError('Surface release prerequisite failed')
  files[key]=p
  if 'input_sha256' in r:pins.update(r['input_sha256'])
  else:
   protocol=p.with_name('protocol.json')
   if sha(protocol)!=r['protocol_sha256']:raise ValueError('Surface release protocol changed')
   pins.update(json.loads(protocol.read_text())['input_sha256']);files[key+'_protocol']=protocol
 if any(sha(p)!=h for p,h in pins.items()):raise ValueError('Surface release prerequisite changed')
 return files
def release_evidence(values,status):
 clear,eligible,bits=map(int,values);rest=struct.unpack('<f',struct.pack('<I',bits))[0] if 0<=bits<2**32 else float('nan')
 checks=dict(host_report=status==0,rest_offset_valid=math.isfinite(rest) and rest>0,counter_consistency=0<=clear<=eligible,actual_changed_history=clear>0)
 return dict(execution_verified=all(checks.values()),checks=checks,cleared_nonzero_sides=clear,additional_envelope_releases=eligible,rest_offset_m=rest if math.isfinite(rest) else None,unsupported_release_distance_m=2*rest if math.isfinite(rest) else None,physical_improvement_accepted=False,performance_accepted=False,platform_acceptance=False)
