"""Explicit, counted endpoint-transition initialization in native cloth history."""
import json
from pathlib import Path
from .cloth_stage_backend import sha
CONTROL='cloth-split-envelope-refresh16-history-birth-control'
BACKEND='cloth-split-envelope-refresh16-edge-query-history-birth-contact'
BACKENDS=(CONTROL,BACKEND)
def artifacts(root):
 from .cloth_history_diagnostic_backend import artifacts as base_artifacts
 root=Path(root);files=base_artifacts(root);build=root/'results/cloth-history-birth-runtime-build-v1';m=json.loads((build/'build.json').read_text())
 if m.get('status')!='compiled' or m.get('conservative_feature_birth') is not True or m.get('birth_branch_counters') is not True:raise ValueError('Explicit counted feature-birth build required')
 files.update(library=build/'libcloth_split_backend.so',manifest=build/'build.json',cloth_split_ptx=build/'contact.ptx')
 if sha(files['library'])!=m['library_sha256'] or sha(files['cloth_split_ptx'])!=m['ptx_sha256']:raise ValueError('Feature birth build changed')
 pins=dict(m['input_sha256'])
 for key,name,count in [('cloth_split_selftest','binding',94),('history_selftest','binding',94),('selftest','dispatch',10),('cache_selftest','cache',12),('history_physics_selftest','physics',640)]:
  p=root/('results/cloth-history-birth-runtime-'+name+'-v1/summary.json');r=json.loads(p.read_text())
  if r.get('passed') is not True or len(r.get('checks',{}))!=count or not all(r['checks'].values()):raise ValueError('Feature birth prerequisite failed')
  files[key]=p
  if 'input_sha256' in r:pins.update(r['input_sha256'])
  else:
   protocol=p.with_name('protocol.json')
   if sha(protocol)!=r['protocol_sha256']:raise ValueError('Feature birth protocol changed')
   pins.update(json.loads(protocol.read_text())['input_sha256']);files[key+'_protocol']=protocol
 if any(sha(p)!=h for p,h in pins.items()):raise ValueError('Feature birth prerequisite changed')
 return files
def birth_evidence(values,status):
 attempted,accepted,rejected=map(int,values)
 complete=status==0 and min(attempted,accepted,rejected)>=0 and attempted==accepted+rejected
 return dict(attempted=attempted,accepted=accepted,rejected=rejected,exit_code=status,report_complete=complete,execution_verified=complete and accepted>0,scope='Counted initialization only. Rejected inactive candidates are not automatically contact errors; history and first-failure evidence must also pass.',platform_acceptance=False)
