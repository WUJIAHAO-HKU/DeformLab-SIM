"""Certified nonzero-side reentry with counted consecutive pre-EE snapshots."""
import json
from pathlib import Path
from .cloth_stage_backend import sha
CONTROL='cloth-split-envelope-refresh16-transport-control'
ABLATION='cloth-split-envelope-refresh16-edge-query-transport-ablation-contact'
BACKEND='cloth-split-envelope-refresh16-edge-query-transport-contact'
BACKENDS={CONTROL:0,ABLATION:1,BACKEND:1}
POLICIES={ABLATION:0,BACKEND:1}

def artifacts(root):
 from .cloth_reentry_runtime_backend import artifacts as base
 root=Path(root);files=base(root);build=root/'results/cloth-history-transport-backend-build-v1';m=json.loads((build/'build.json').read_text())
 if m.get('status')!='compiled' or not all(m.get(k) is True for k in ('retained_side_transport','consecutive_preEE_snapshots','transport_policy')):raise ValueError('Verified consecutive-history transport build required')
 files.update(library=build/'libcloth_split_backend.so',manifest=build/'build.json',cloth_split_ptx=build/'contact.ptx')
 pins={**m['input_sha256'],str(files['library']):m['library_sha256'],str(files['cloth_split_ptx']):m['ptx_sha256']}
 for key,folder,count in [('cloth_split_selftest','cloth-history-transport-binding-v1',461),('history_selftest','cloth-history-transport-binding-v1',461),('transport_sequence_selftest','cloth-history-transport-sequence-v1',50),('transport_probe_selftest','cloth-history-transport-probe-v1',427),('selftest','cloth-history-transport-dispatch-v1',10),('cache_selftest','cloth-history-transport-cache-v1',12),('history_physics_selftest','cloth-history-transport-physics-v1',640)]:
  p=root/'results'/folder/'summary.json';r=json.loads(p.read_text())
  if r.get('passed') is not True or len(r.get('checks',{}))!=count or not all(r['checks'].values()):raise ValueError('Transport prerequisite failed: '+folder)
  files[key]=p
  if 'input_sha256' in r:pins.update(r['input_sha256'])
  else:
   protocol=p.with_name('protocol.json')
   if sha(protocol)!=r['protocol_sha256']:raise ValueError('Transport protocol changed')
   pins.update(json.loads(protocol.read_text())['input_sha256']);files[key+'_protocol']=protocol
 if any(sha(p)!=h for p,h in pins.items()):raise ValueError('Transport prerequisite changed')
 return files

def evidence(policy,values,status,policy_values,policy_status,epochs,nodes):
 if policy not in (0,1) or type(policy) is not int or len(values)!=10:raise ValueError('Explicit transport policy and ten counters required')
 attempt,accepted,retained,visited,maximum,bad,copies,copied_nodes,copy_errors,last_epoch=map(int,values)
 checks=dict(host_reports=status==policy_status==0,explicit_device_policy=list(policy_values)==[policy,1,policy],
  counter_partition=0<=accepted<=attempt and 0<=retained and attempt==accepted+retained,
  bounded_screen=0<=maximum<=65 and 0<=visited<=65*attempt and visited>=maximum,
  valid_snapshot_inputs=bad==0,consecutive_snapshots=epochs>0 and nodes>0 and copies==epochs+1 and copied_nodes==(epochs+1)*nodes and copy_errors==0 and last_epoch==epochs,
  actual_branch_execution=accepted>0 if policy else attempt==accepted==retained==visited==maximum==0)
 return dict(execution_verified=all(checks.values()),checks=checks,policy=policy,attempts=attempt,certified_retained_transitions=accepted,conservative_retained_transitions=retained,
  copied_snapshots=copies,copied_nodes=copied_nodes,last_snapshot_epoch=last_epoch,additional_geometry_storage_bytes=16*nodes,
  physical_improvement_accepted=False,performance_accepted=False,platform_acceptance=False,
  scope='Consecutive pre-EE finite-separation screen for retained nonzero sides only on unsupported-to-interior reentry. Actual crossed or unresolved chords retain the old side. Policy 0 is an explicit transport ablation; zero-side initialization and other contact repairs remain enabled.')
