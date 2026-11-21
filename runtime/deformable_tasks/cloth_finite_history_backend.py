"""Explicit, counted endpoint-transition initialization in native cloth history."""
import json
from pathlib import Path
from .cloth_stage_backend import sha
CONTROL='cloth-split-envelope-refresh16-history-finite-control'
BACKEND='cloth-split-envelope-refresh16-edge-query-history-finite-contact'
BACKENDS=(CONTROL,BACKEND)
def artifacts(root):
 from .cloth_history_birth_backend import artifacts as base_artifacts
 root=Path(root);files=base_artifacts(root);build=root/'results/cloth-finite-history-runtime-build-v1';m=json.loads((build/'build.json').read_text())
 if m.get('status')!='compiled' or m.get('finite_segment_birth') is not True or m.get('finite_branch_counters') is not True:raise ValueError('Explicit counted feature-birth build required')
 files.update(library=build/'libcloth_split_backend.so',manifest=build/'build.json',cloth_split_ptx=build/'contact.ptx')
 if sha(files['library'])!=m['library_sha256'] or sha(files['cloth_split_ptx'])!=m['ptx_sha256']:raise ValueError('Feature birth build changed')
 pins=dict(m['input_sha256'])
 for key,name,count in [('cloth_split_selftest','binding',128),('history_selftest','binding',128),('selftest','dispatch',10),('cache_selftest','cache',12),('history_physics_selftest','physics',640)]:
  p=root/('results/cloth-finite-history-runtime-'+name+'-v1/summary.json');r=json.loads(p.read_text())
  if r.get('passed') is not True or len(r.get('checks',{}))!=count or not all(r['checks'].values()):raise ValueError('Feature birth prerequisite failed')
  files[key]=p
  if 'input_sha256' in r:pins.update(r['input_sha256'])
  else:
   protocol=p.with_name('protocol.json')
   if sha(protocol)!=r['protocol_sha256']:raise ValueError('Feature birth protocol changed')
   pins.update(json.loads(protocol.read_text())['input_sha256']);files[key+'_protocol']=protocol
 for key,folder,count in [('finite_geometry_selftest','cloth-finite-history-verification-v3',1343),('finite_cache_selftest','cloth-finite-history-cache-v1',62)]:
  p=root/'results'/folder/'summary.json';r=json.loads(p.read_text())
  if r.get('passed') is not True or len(r.get('checks',{}))!=count or not all(r['checks'].values()):raise ValueError('Finite geometry/cache prerequisite failed')
  files[key]=p
  if 'input_sha256' in r:pins.update(r['input_sha256'])
  else:
   protocol=p.with_name('protocol.json')
   if sha(protocol)!=r['protocol_sha256']:raise ValueError('Finite cache protocol changed')
   pins.update(json.loads(protocol.read_text())['input_sha256']);files[key+'_protocol']=protocol
 if any(sha(p)!=h for p,h in pins.items()):raise ValueError('Feature birth prerequisite changed')
 return files
def finite_evidence(values,status,birth):
 attempted,accepted,rejected,total,maximum=map(int,values)
 complete=status==0 and min(attempted,accepted,rejected,total,maximum)>=0 and attempted==accepted+rejected and maximum<=65 and maximum<=total<=65*attempted
 complete &= (total==maximum==0 if attempted==0 else maximum>0)
 complete &= attempted<=birth.get('attempted',-1) and accepted<=birth.get('accepted',-1) and rejected<=birth.get('rejected',-1)
 return dict(attempted=attempted,accepted=accepted,rejected=rejected,total_intervals=total,max_intervals=maximum,exit_code=status,report_complete=bool(complete),execution_verified=bool(complete and accepted>0),scope='Bounded finite-segment initialization, not actual-trajectory CCD or performance acceptance.',platform_acceptance=False)
