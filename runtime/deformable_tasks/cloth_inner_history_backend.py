"""Explicit, counted endpoint-transition initialization in native cloth history."""
import json
from pathlib import Path
from .cloth_stage_backend import sha
BACKEND='cloth-split-envelope-refresh16-edge-query-history-inner-observe'
TARGETS=((21,43,64,85),(607,628,608,629))
KEYS=tuple(sum(n<<(48-16*j) for j,n in enumerate(ids)) for ids in TARGETS)
def artifacts(root):
 from .cloth_finite_history_backend import artifacts as base_artifacts
 root=Path(root);files=base_artifacts(root);build=root/'results/cloth-inner-history-build-v2';m=json.loads((build/'build.json').read_text())
 if m.get('status')!='compiled' or m.get('inner_epoch_trace') is not True or m.get('trace_stride_words')!=48:raise ValueError('Explicit counted feature-birth build required')
 files.update(library=build/'libcloth_split_backend.so',manifest=build/'build.json',cloth_split_ptx=build/'contact.ptx')
 if sha(files['library'])!=m['library_sha256'] or sha(files['cloth_split_ptx'])!=m['ptx_sha256']:raise ValueError('Feature birth build changed')
 pins=dict(m['input_sha256'])
 for key,name,count in [('cloth_split_selftest','binding',202),('history_selftest','binding',202),('selftest','dispatch',10),('cache_selftest','cache',12),('history_physics_selftest','physics',640)]:
  p=root/('results/cloth-inner-history-'+name+'-v2/summary.json');r=json.loads(p.read_text())
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
def trace_evidence(path,values,status,steps,calls):
 import numpy as np
 path=Path(path);epochs,targets,records,overflow,invalid,io_error=map(int,values)
 checks=dict(host_report=status==0 and io_error==overflow==invalid==0,epoch_coverage=epochs==calls and records==2*calls and targets==2,file_present=path.exists())
 detail=[];digest=None
 if path.exists():
  raw=np.fromfile(path,dtype='<u4');digest=sha(path)
  checks['byte_count']=len(raw)==12+epochs*2*48 and path.stat().st_size==4*(12+epochs*2*48)
  if checks['byte_count'] and len(raw)>=12:
   h=raw[:12];a=raw[12:].reshape(epochs,2,48)
   checks['header']=list(map(int,h[:8]))==[0x46544831,1,2,131072,epochs,48,records,0] and tuple(int(h[8+2*k])+(int(h[9+2*k])<<32) for k in range(2))==KEYS
   checks['packets_complete']=bool(np.all(a[:,:,0]==2))
   checks['packet_target_identity']=all(bool(np.all(a[:,k,4]==(KEYS[k]&0xffffffff)) and np.all(a[:,k,5]==(KEYS[k]>>32)) and np.all(a[:,k,15]>max(TARGETS[k]))) for k in range(2))
   checks['ordered_epochs']=bool(np.all(a[:,:,1]==np.arange(1,epochs+1)[:,None]))
   checks['finite_positions']=bool(np.isfinite(a[:,:,16:48].view('<f4')).all())
   checks['candidate_flags']=bool(np.all(a[:,:,7]>=a[:,:,8]) and np.all(a[:,:,8]>=a[:,:,9]))
   checks['frame_coverage']=epochs>0 and steps>0 and epochs%steps==0 and bool(np.all(a[:,:,2]==(np.arange(epochs)//(epochs//steps)+1)[:,None])) and bool(np.all(a[:,:,3]==0))
   if checks['frame_coverage']:
    b=a[:,:,32:48].reshape(steps,epochs//steps,2,16);checks['owned_frame_history_constant']=bool(np.all(b==b[:,0:1]))
   for k in range(2):
    r=a[:,k]
    def first(mask):
     indices=np.flatnonzero(mask)
     return int(r[indices[0],1]) if len(indices) else None
    detail.append(dict(nodes=list(TARGETS[k]),first_raw_candidate_epoch=first(r[:,7]>0),first_enabled_epoch=first(r[:,8]>0),first_active_epoch=first(r[:,9]>0),first_ready_history_epoch=first((r[:,11]==1)|(r[:,11]==5))))
 return dict(execution_verified=all(checks.values()),checks=checks,epochs=epochs,targets=detail,records=records,file=str(path),sha256=digest,allocated_trace_bytes=131072*2*48*4+48,read_only=True,platform_acceptance=False)
