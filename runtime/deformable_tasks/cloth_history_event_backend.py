"""Explicit event-ordered history route; same physical PTX, independent native binding."""
import json
from pathlib import Path
from .cloth_stage_backend import sha
CONTROL='cloth-split-envelope-refresh16-history-events-control'
BACKEND='cloth-split-envelope-refresh16-edge-query-history-events-contact'
BACKENDS=(CONTROL,BACKEND)
def artifacts(root):
 from .cloth_history_backend import artifacts as base_artifacts
 root=Path(root);files=base_artifacts(root);physical=files['cloth_split_ptx'];build=root/'results/cloth-history-event-build-v2';m=json.loads((build/'build.json').read_text())
 if m.get('event_ordered_history') is not True or m.get('status')!='compiled' or sha(physical)!=m['ptx_sha256']:raise ValueError('Matching physical module and event binding required')
 files.update(library=build/'libcloth_split_backend.so',manifest=build/'build.json',cloth_split_ptx=build/'contact.ptx')
 if sha(files['library'])!=m['library_sha256'] or sha(files['cloth_split_ptx'])!=m['ptx_sha256']:raise ValueError('Event build changed')
 pins=dict(m['input_sha256'])
 for key,name,count in [('cloth_split_selftest','binding',50),('history_selftest','binding',50),('selftest','dispatch',10),('cache_selftest','cache',12)]:
  p=root/('results/cloth-history-event-'+name+'-v1/summary.json');r=json.loads(p.read_text())
  if r.get('passed') is not True or len(r.get('checks',{}))!=count or not all(r['checks'].values()):raise ValueError('Event prerequisite failed')
  files[key]=p;pins.update(r['input_sha256'])
 if any(sha(p)!=h for p,h in pins.items()):raise ValueError('Event prerequisite changed')
 return files

def stream_evidence(values,status,steps,counts):
 v=list(map(int,values));solves=counts.get('cloth_solveClothClothDeltaVTLaunch',0)+counts.get('cloth_solveClothClothDeltaEELaunch',0);queries=counts.get('cloth_queryClothClothContactEECountLaunch',0)
 valid=len(v)==5 and status==0 and v[0]==steps>0 and v[1]==solves>0 and v[2]>=solves+queries and v[3]==v[4]==2
 return dict(snapshot_events=v[0],publication_events=v[1],stream_waits=v[2],events_created=v[3],events_destroyed=v[4],exit_code=status,execution_verified=bool(valid),global_context_sync_per_solve=False,platform_acceptance=False)
