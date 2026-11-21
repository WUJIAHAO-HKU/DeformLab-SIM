"""Counted finite-edge initialization for both new and released contact features."""
import json
from pathlib import Path
from .cloth_stage_backend import sha
BACKEND='cloth-split-envelope-refresh16-edge-query-history-surface-reentry-contact'
def artifacts(root):
 from .cloth_surface_release_backend import artifacts as base
 root=Path(root);files=base(root);build=root/'results/cloth-surface-reentry-backend-build-v3';m=json.loads((build/'build.json').read_text())
 if m.get('status')!='compiled' or m.get('certified_finite_initialization') is not True or m.get('all_feature_failure_ring') is not True:raise ValueError('Certified finite initialization with failure observation required')
 files.update(library=build/'libcloth_split_backend.so',manifest=build/'build.json',cloth_split_ptx=build/'contact.ptx')
 if sha(files['library'])!=m['library_sha256'] or sha(files['cloth_split_ptx'])!=m['ptx_sha256']:raise ValueError('Finite initialization build changed')
 pins=dict(m['input_sha256'])
 for key,folder,count in [('cloth_split_selftest','cloth-surface-reentry-binding-v4',394),('history_selftest','cloth-surface-reentry-binding-v4',394),('selftest','cloth-surface-reentry-dispatch-v3',10),('cache_selftest','cloth-surface-reentry-cache-v3',12),('history_physics_selftest','cloth-surface-reentry-physics-v3',640),('finite_initialization_witness0_selftest','cloth-surface-reentry-probe-v3-block0',409),('finite_initialization_witness2_selftest','cloth-surface-reentry-probe-v3-block2',409),('finite_initialization_cpu_selftest','cloth-surface-reentry-cpu-v3',89)]:
  p=root/'results'/folder/'summary.json';r=json.loads(p.read_text())
  if r.get('passed') is not True or len(r.get('checks',{}))!=count or not all(r['checks'].values()):raise ValueError('Finite initialization prerequisite failed: '+folder)
  files[key]=p
  if 'input_sha256' in r:pins.update(r['input_sha256'])
  else:
   protocol=p.with_name('protocol.json')
   if sha(protocol)!=r['protocol_sha256']:raise ValueError('Finite initialization protocol changed')
   pins.update(json.loads(protocol.read_text())['input_sha256']);files[key+'_protocol']=protocol
 if any(sha(p)!=h for p,h in pins.items()):raise ValueError('Finite initialization prerequisite changed')
 return files

def initialization_evidence(values,status):
 attempts,certified,retained=map(int,values)
 checks=dict(host_report=status==0,nonnegative_counts=min(attempts,certified,retained)>=0,counter_partition=attempts==certified+retained,actual_certified_reorientation=certified>0)
 return dict(execution_verified=all(checks.values()),checks=checks,opposite_root_side_attempts=attempts,certified_finite_reorientations=certified,root_side_corrections_retained=retained,physical_improvement_accepted=False,performance_accepted=False,platform_acceptance=False,scope='Counts only zero-side initialization. Unresolved/actual crossing certificates retain the existing root-side correction; a retained count is not a runtime error. Nonzero side histories are unchanged.')

def finite_budget_evidence(values,status,birth,initialization):
 from .cloth_finite_history_backend import finite_evidence
 pool=dict(attempted=birth.get('attempted',-1)+initialization.get('opposite_root_side_attempts',-1),accepted=birth.get('accepted',-1)+initialization.get('certified_finite_reorientations',-1),rejected=birth.get('rejected',-1)+initialization.get('root_side_corrections_retained',-1))
 result=finite_evidence(values,status,pool)
 counters=initialization.get('checks',{})
 complete=birth.get('report_complete') is True and all(counters.get(k) is True for k in ('host_report','nonnegative_counts','counter_partition'))
 result['report_complete'] &= complete;result['execution_verified'] &= complete
 result['counter_scope']='Finite screen is shared by unsupported-root births and opposite-valid-root initializations; counts are bounded by their disjoint sum.'
 result['initialization_pool']=pool
 return result
