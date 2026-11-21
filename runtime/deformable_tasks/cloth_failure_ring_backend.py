"""Read-only all-feature pre-failure history observation; never qualifies physics."""
import json
from pathlib import Path
from .cloth_stage_backend import sha
BACKEND='cloth-split-envelope-refresh16-edge-query-history-ring-observe'
def artifacts(root):
 from .cloth_inner_history_backend import artifacts as base
 root=Path(root);files=base(root);build=root/'results/cloth-failure-ring-backend-build-v1';m=json.loads((build/'build.json').read_text())
 if m.get('status')!='compiled' or m.get('all_feature_failure_ring') is not True or m.get('ring_slots')!=128:raise ValueError('Explicit failure ring build required')
 files.update(library=build/'libcloth_split_backend.so',manifest=build/'build.json',cloth_split_ptx=build/'contact.ptx')
 if sha(files['library'])!=m['library_sha256'] or sha(files['cloth_split_ptx'])!=m['ptx_sha256']:raise ValueError('Failure ring build changed')
 pins=dict(m['input_sha256'])
 for key,name,count in [('cloth_split_selftest','binding',240),('history_selftest','binding',240),('selftest','dispatch',10),('cache_selftest','cache',12),('history_physics_selftest','physics',640),('failure_ring_selftest','probe-verification',45)]:
  p=root/('results/cloth-failure-ring-'+name+'-v1/summary.json');r=json.loads(p.read_text())
  if r.get('passed') is not True or len(r.get('checks',{}))!=count or not all(r['checks'].values()):raise ValueError('Failure ring prerequisite failed')
  files[key]=p
  if 'input_sha256' in r:pins.update(r['input_sha256'])
  else:
   protocol=p.with_name('protocol.json')
   if sha(protocol)!=r['protocol_sha256']:raise ValueError('Failure ring protocol changed')
   pins.update(json.loads(protocol.read_text())['input_sha256']);files[key+'_protocol']=protocol
 if any(sha(p)!=h for p,h in pins.items()):raise ValueError('Failure ring prerequisite changed')
 return files

def ring_evidence(path,values,status,steps,calls,diagnostics):
 import numpy as np
 path=Path(path);slots,nodes,capacity,captures,published,skipped,errors,allocated,io=map(int,values)
 failure=diagnostics.get('first_rejection');decode=diagnostics.get('first_decode_error')
 end=min([calls,*[int(v['epoch']) for v in (failure,decode) if v]])
 checks=dict(host_report=status==0 and errors==io==0,dimensions=slots==128 and 0<nodes<=4096 and 0<capacity<=32768,coverage=captures==published==end and skipped==calls-end,diagnostics_complete=diagnostics.get('report_complete') is True,file_present=path.exists(),memory_accounted=allocated==slots*(32+nodes*32+capacity*24)+32)
 digest=None;begin=max(1,end-slots+1)
 if path.exists():
  digest=sha(path);checks['byte_count']=path.stat().st_size==64+allocated
  if checks['dimensions'] and checks['byte_count']:
   h=np.fromfile(path,dtype='<u4',count=16);counter=np.fromfile(path,dtype='<u8',count=4,offset=64)
   checks['header']=list(h)==[0x46524831,1,slots,nodes,capacity,6,8,4,calls,failure['epoch'] if failure else 0,decode['epoch'] if decode else 0,0,0,0,0,0]
   checks['counter_bytes']=list(counter)==[captures,published,skipped,errors]
   raw=np.memmap(path,dtype='<u4',mode='r',offset=96);meta=raw[:slots*8].reshape(slots,8);off=slots*8
   p=raw[off:off+slots*nodes*4].view('<f4').reshape(slots,nodes,4);off+=slots*nodes*4
   roots=raw[off:off+slots*nodes*4].view('<f4').reshape(slots,nodes,4);off+=slots*nodes*4
   cache=raw[off:].reshape(slots,capacity,6)
   epochs=np.arange(begin,end+1,dtype=np.uint32);indices=(epochs-1)%slots
   checks['published_epochs']=bool(len(epochs)>0 and np.all(meta[indices,0]==2) and np.array_equal(meta[indices,1],epochs) and np.all(meta[indices,4]==nodes) and np.all(meta[indices,5]==capacity) and np.all(meta[indices,6]==1) and np.all(meta[indices,7]==0))
   checks['frames']=calls>0 and steps>0 and calls%steps==0 and bool(np.all(meta[indices,2]==(epochs-1)//(calls//steps)+1) and np.all(meta[indices,3]==0))
   checks['finite_positions']=bool(np.isfinite(p[indices]).all() and np.isfinite(roots[indices]).all())
   checks['history_epochs']=bool(np.all(cache[indices,:,3]<=epochs[:,None]) and np.all(cache[indices,:,5]<=epochs[:,None]))
   if end<slots:
    checks['unused_reset']=not bool(meta[end:].any() or p[end:].any() or roots[end:].any() or cache[end:].any())
   if failure and begin<=failure['epoch']<=end:
    i=(failure['epoch']-1)%slots;ids=failure['node_ids'];at=failure['slot']
    valid=all(0<=n<nodes for n in ids) and 0<=at<capacity
    checks['first_rejection_snapshot']=valid and bool(np.array_equal(p[i,ids],np.array(failure['current_position_inverse_mass'],dtype='<f4')) and np.array_equal(roots[i,ids],np.array(failure['owned_previous_position_inverse_mass'],dtype='<f4')) and list(cache[i,at])==[failure['key']&0xffffffff,failure['key']>>32,failure['side']&0xffffffff,failure['history_epoch'],failure['history_state'],failure['last_seen']])
 return dict(execution_verified=all(checks.values()),checks=checks,file=str(path),sha256=digest,first_retained_epoch=begin,last_retained_epoch=end,allocated_bytes=allocated,read_only=True,platform_acceptance=False,scope='All nodal positions and history slots at pre-EE boundaries. Enabled arrival is inferred only from cache seen epochs; raw/active row census and individual intervening kernel paths are not captured.')
