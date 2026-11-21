"""Explicit persistent EE history route; physical/cost acceptance remains separate."""
import json
from pathlib import Path
from .cloth_stage_backend import sha
CONTROL='cloth-split-envelope-refresh16-history-control'
BACKEND='cloth-split-envelope-refresh16-edge-query-history-contact'
BACKENDS=(CONTROL,BACKEND)
def artifacts(root):
 from .cloth_edge_query_backend import artifacts as base_artifacts
 root=Path(root);files=base_artifacts(root);build=root/'results/cloth-history-runtime-build-v1';m=json.loads((build/'build.json').read_text())
 if m.get('status')!='compiled' or m.get('explicit_history_activation') is not True or m.get('device_indirect_row_count') is not True:raise ValueError('Explicit native history build required')
 files.update(library=build/'libcloth_split_backend.so',manifest=build/'build.json',cloth_split_ptx=build/'contact.ptx')
 if sha(files['library'])!=m['library_sha256'] or sha(files['cloth_split_ptx'])!=m['ptx_sha256']:raise ValueError('History build changed')
 pins=dict(m['input_sha256'])
 for key,name,count in [('cloth_split_selftest','binding',47),('history_selftest','binding',47),('selftest','dispatch',10),('cache_selftest','cache',12),('history_physics_selftest','physics',640)]:
  p=root/('results/cloth-history-runtime-'+name+'-v1/summary.json');r=json.loads(p.read_text())
  if r.get('passed') is not True or len(r.get('checks',{}))!=count or not all(r['checks'].values()):raise ValueError('History prerequisite failed: '+name)
  files[key]=p
  if 'input_sha256' in r:pins.update(r['input_sha256'])
  else:
   protocol=p.with_name('protocol.json')
   if sha(protocol)!=r['protocol_sha256']:raise ValueError('History test protocol changed')
   pins.update(json.loads(protocol.read_text())['input_sha256']);files[key+'_protocol']=protocol
 if any(sha(p)!=h for p,h in pins.items()):raise ValueError('History prerequisite changed')
 return files

def evidence(values,status,steps,counts,*,position_snapshots=1):
 if type(position_snapshots) is not int or position_snapshots not in (1,2):raise ValueError("Explicit supported history snapshot layout required")
 v=list(map(int,values));expected=counts.get('cloth_solveClothClothDeltaEELaunch',0)
 valid=len(v)==16 and status==0 and v[0]==v[8]==steps and steps>0 and v[1]==v[7]==expected and expected>0 and v[2]==v[3]+v[4]+v[5] and v[5]>0 and v[6]==v[9]==v[10]==v[15]==0 and 3<=v[11]<=65536 and v[12]>0 and v[13]==32768 and v[14]==24*v[13]+16*v[11]*position_snapshots+400000+68
 return dict(frames=v[0],host_updates=v[1],candidate_rows=v[2],disabled_rows=v[3],shared_node_rows=v[4],resolved_rows=v[5],decode_errors=v[6],gpu_resolve_calls=v[7],gpu_frame_captures=v[8],history_errors=v[10],nodes=v[11],triangles=v[12],capacity=v[13],scratch_bytes=v[14],position_snapshots=position_snapshots,expected_scratch_bytes=24*v[13]+16*v[11]*position_snapshots+400000+68,read_errors=v[15],exit_code=status,execution_verified=bool(valid),ee_history_only=True,previous_root_owned=True,physical_improvement_accepted=False,performance_accepted=False,platform_acceptance=False)
