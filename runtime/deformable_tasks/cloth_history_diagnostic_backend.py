"""First-rejection observation for the explicit event-ordered EE history backend."""
import json,struct
from pathlib import Path
from .cloth_stage_backend import sha
BACKEND='cloth-split-envelope-refresh16-edge-query-history-diagnostic-contact'
def artifacts(root):
 from .cloth_history_event_backend import artifacts as base_artifacts
 root=Path(root);files=base_artifacts(root);build=root/'results/cloth-history-diagnostic-build-v2';m=json.loads((build/'build.json').read_text())
 if m.get('status')!='compiled' or m.get('first_failure_diagnostics') is not True:raise ValueError('Explicit first-failure build required')
 files.update(library=build/'libcloth_split_backend.so',manifest=build/'build.json',cloth_split_ptx=build/'contact.ptx')
 if sha(files['library'])!=m['library_sha256'] or sha(files['cloth_split_ptx'])!=m['ptx_sha256']:raise ValueError('Diagnostic build changed')
 pins=dict(m['input_sha256'])
 for key,name,count in [('cloth_split_selftest','binding',60),('history_selftest','binding',60),('selftest','dispatch',10),('cache_selftest','cache',12),('history_physics_selftest','physics',640)]:
  p=root/('results/cloth-history-diagnostic-'+name+'-v1/summary.json');r=json.loads(p.read_text())
  if r.get('passed') is not True or len(r.get('checks',{}))!=count or not all(r['checks'].values()):raise ValueError('Diagnostic prerequisite failed')
  files[key]=p
  if 'input_sha256' in r:pins.update(r['input_sha256'])
  else:
   protocol=p.with_name('protocol.json')
   if sha(protocol)!=r['protocol_sha256']:raise ValueError('Diagnostic protocol changed')
   pins.update(json.loads(protocol.read_text())['input_sha256']);files[key+'_protocol']=protocol
 if any(sha(p)!=h for p,h in pins.items()):raise ValueError('Diagnostic prerequisite changed')
 return files

def failure_evidence(failure,decode,status):
 f=list(map(int,failure));d=list(map(int,decode));record=None;decoded=None
 def quad(start):
  values=struct.unpack('<16f',struct.pack('<16I',*f[start:start+16]));return [list(values[i:i+4]) for i in range(0,16,4)]
 if f[0]==2:
  record=dict(reason=f[1],phase={0:'normal',1:'friction',2:'weight_query'}.get(f[2],str(f[2])),row=f[3],epoch=f[4],nodes=f[5],encoded_row=f[6:10],slot=f[10],side=struct.unpack('<i',struct.pack('<I',f[11]))[0],history_state=f[12],history_epoch=f[13],last_seen=f[14],key=f[15]+(f[16]<<32),triangles=f[17],node_ids=f[18:22],current_position_inverse_mass=quad(24),owned_previous_position_inverse_mass=quad(40),native_previous_position_inverse_mass=quad(56),frame=f[72]+(f[73]<<32),rest_offset_m=struct.unpack('<f',struct.pack('<I',f[74]))[0],h_s=struct.unpack('<f',struct.pack('<I',f[75]))[0],edge=bool(f[76]))
 if d[0]==2:decoded=dict(reason=d[1],row=d[2],count=d[3],epoch=d[4],encoded_row=d[5:9],frame=d[9])
 return dict(report_complete=status==0 and f[0] in (0,2) and d[0] in (0,2),first_rejection=record,first_decode_error=decoded,raw_failure_words=f,raw_decode_words=d,exit_code=status,read_only=True,platform_acceptance=False)
