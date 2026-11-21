"""Same-binary native/ablation/reentry routes without full history recording."""
import json
from pathlib import Path
from .cloth_stage_backend import sha
CONTROL='cloth-split-envelope-refresh16-reentry-runtime-control'
ABLATION='cloth-split-envelope-refresh16-edge-query-reentry-runtime-ablation-contact'
BACKEND='cloth-split-envelope-refresh16-edge-query-reentry-runtime-contact'
BACKENDS={CONTROL:0,ABLATION:1,BACKEND:1}
POLICIES={ABLATION:0,BACKEND:1}

def artifacts(root):
 from .cloth_surface_reentry_backend import artifacts as base
 root=Path(root);files=base(root);build=root/'results/cloth-reentry-policy-backend-build-v1';m=json.loads((build/'build.json').read_text())
 if m.get('status')!='compiled' or m.get('runtime_initialization_policy') is not True or m.get('trajectory_observer') is not False or m.get('transition_only_release_counts') is not True:raise ValueError('Verified same-binary policy runtime required')
 files.update(library=build/'libcloth_split_backend.so',manifest=build/'build.json',cloth_split_ptx=build/'contact.ptx')
 pins={**m['input_sha256'],str(files['library']):m['library_sha256'],str(files['cloth_split_ptx']):m['ptx_sha256']}
 for key,folder,count in [('cloth_split_selftest','cloth-reentry-policy-binding-v1',366),('history_selftest','cloth-reentry-policy-binding-v1',366),('selftest','cloth-reentry-policy-dispatch-v1',10),('cache_selftest','cloth-reentry-policy-cache-v1',12),('history_physics_selftest','cloth-reentry-policy-physics-v1',640),('finite_initialization_witness0_selftest','cloth-reentry-policy-probe-block0-v1',409),('finite_initialization_witness2_selftest','cloth-reentry-policy-probe-block2-v1',409)]:
  p=root/'results'/folder/'summary.json';r=json.loads(p.read_text())
  if r.get('passed') is not True or len(r.get('checks',{}))!=count or not all(r['checks'].values()):raise ValueError('Policy runtime prerequisite failed: '+folder)
  files[key]=p
  if 'input_sha256' in r:pins.update(r['input_sha256'])
  else:
   protocol=p.with_name('protocol.json')
   if sha(protocol)!=r['protocol_sha256']:raise ValueError('Policy protocol changed')
   pins.update(json.loads(protocol.read_text())['input_sha256']);files[key+'_protocol']=protocol
 if any(sha(p)!=h for p,h in pins.items()):raise ValueError('Policy runtime prerequisite changed')
 return files

def policy_evidence(requested,values,status):
 if requested not in (0,1) or type(requested) is not int:raise ValueError('Explicit initialization policy required')
 checks=dict(host_report=status==0,requested_configured_and_device_policy=list(values)==[requested,1,requested])
 return dict(execution_verified=all(checks.values()),checks=checks,requested=requested,values=list(values),same_binary=True,physical_improvement_accepted=False,performance_accepted=False,platform_acceptance=False)

def initialization_evidence(policy,values,status):
 from .cloth_surface_reentry_backend import initialization_evidence as active
 if policy not in (0,1) or type(policy) is not int:raise ValueError('Explicit initialization policy required')
 result=active(values,status)
 if policy==0:
  checks=dict(host_report=status==0,nonnegative_counts=min(values)>=0,counter_partition=values[0]==values[1]+values[2],disabled_branch_not_executed=list(values)==[0,0,0])
  result.update(execution_verified=all(checks.values()),checks=checks,ablation=True,scope='Explicit policy-0 ablation: opposite-root finite reorientation must not execute. Other history/contact repairs remain active. This is not native physics and provides no proof that the disabled repair improves a task.')
 else:result['ablation']=False
 return result
