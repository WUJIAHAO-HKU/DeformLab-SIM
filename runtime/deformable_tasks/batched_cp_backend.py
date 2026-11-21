"""Named restricted CP mapping with device coverage and frame-boundary validation."""
import ctypes,json,os
from pathlib import Path
from .cone_backend import sha
from .batched_geometry_backend import BACKENDS as GEOMETRY_BACKENDS
BACKENDS={'volume-batched-control':0,'volume-batched-mapping':1,**GEOMETRY_BACKENDS}
def requested(task,backend):
 if backend not in BACKENDS:return False
 if task!='multi_soft':raise ValueError('Batched CP supports exactly two volume bodies in multi_soft; no fallback')
 if any(os.environ.get(k,'0')!='0' for k in ('DEFORMABLE_CP_MAPPING','DEFORMABLE_CONE_TRACE','DEFORMABLE_KERNEL_PROFILE','DEFORMABLE_MULTISOFT_STATES','DEFORMABLE_MULTISOFT_HISTORY','DEFORMABLE_CLOTH_STAGES','DEFORMABLE_CLOTH_SPLIT_STAGES')):raise ValueError('Named batched CP requires its own observer path without mixed diagnostic switches')
 return True

def artifacts(root,backend="volume-batched-mapping"):
 root=Path(root);build=root/'results/batched-cp-binding-build-v1'
 files=dict(library=build/'libbatched_cp_mapping.so',manifest=build/'build.json',batch_ptx=root/'results/batched-cp-mapping-build-v1/batch-mapping.ptx',
  batch_primitive=root/'results/batched-cp-mapping-owned-v1/summary.json',batch_selftest=root/'results/batched-cp-binding-owned-v1/summary.json',batch_native_average=root/'results/batched-cp-native-average-v1/summary.json',batch_lifecycle=root/'results/batched-cp-lifecycle-v1/summary.json')
 for key in ('manifest','batch_primitive','batch_selftest','batch_native_average','batch_lifecycle'):
  record=json.loads(files[key].read_text())
  if key=='manifest':
   if record['status']!='compiled' or sha(files['library'])!=record['library_sha256'] or Path(record['helper_ptx'])!=files['batch_ptx']:raise ValueError('Batched CP build mismatch')
  elif record.get('passed') is not True:raise ValueError('Batched CP prerequisite failed')
  for p,h in record['input_sha256'].items():
   if sha(p)!=h:raise ValueError('Changed batched CP prerequisite '+p)
 if backend in GEOMETRY_BACKENDS:
  from .batched_geometry_backend import extend_artifacts
  files=extend_artifacts(root,files)
 return files

class BatchedCP:
 def __init__(self,lib,files,backend,dest=None):
  self.lib=lib;self.mode=BACKENDS[backend];self.stopped=False
  lib.batched_cp_start.argtypes=[ctypes.c_int,ctypes.c_char_p];lib.batched_cp_stop.argtypes=[ctypes.POINTER(ctypes.c_ulonglong)]
  self.geometry=None;self.geometry_result=None
  if lib.batched_cp_start(self.mode,str(files['batch_ptx']).encode()):raise RuntimeError('Cannot start batched CP mapping')
  if 'geometry_ptx' in files:
   from .batched_geometry_backend import GeometryObserver
   try:
    if dest is None:raise ValueError('Geometry evidence destination required')
    self.geometry=GeometryObserver(lib,files,Path(dest)/'batched-geometry')
   except BaseException:
    values=(ctypes.c_ulonglong*17)();lib.batched_cp_stop(values);raise
 def stop(self):
  if self.stopped:raise RuntimeError('Batched CP stopped twice')
  if self.geometry:self.geometry_result=self.geometry.stop()
  values=(ctypes.c_ulonglong*17)();status=self.lib.batched_cp_stop(values);self.stopped=True
  return self.mode,list(values),status

def evidence(mode,values,status,steps,native):
 if mode not in (0,1) or len(values)!=17:raise ValueError('Unknown batched CP receipt ABI')
 frames,cycles,parts,published,errors,bodies,scratch,syncs=map(int,values[:8]);gpu=list(map(int,values[8:16]));rollbacks=int(values[16]);nodes=scratch//112
 checks=dict(clean=status==errors==0,body_count=bodies==2,scratch_layout=scratch>0 and scratch%112==0,full_steps=frames==steps and steps>0,
  cycles=cycles==published==native.get('sb_gm_cp_averageVertsLaunch',0) and cycles>0,
  partitions=parts==native.get('sb_gm_cp_solveTetrahedronsPartitionLaunch',0) and parts>=cycles,
  preintegrate=frames==native.get('sb_gm_preIntegrateLaunch',0),finalize=frames==native.get('sb_gm_finalizeVelocitiesLaunch',0),frame_validation=syncs==frames,
  gpu_nodes=gpu[:3]==[nodes*cycles]*3,gpu_bodies=gpu[3:6]==[2*cycles]*3,gpu_finite=gpu[6]==0,gpu_validation=gpu[7]==nodes*cycles,no_rollback=rollbacks==0)
 verified=all(checks.values())
 return dict(mode=mode,frames=frames,cycles=cycles,partition_calls=parts,publication_calls=published,errors=errors,bodies=bodies,total_nodes=nodes,scratch_bytes=scratch,frame_synchronizations=syncs,gpu_counters=gpu,rollbacks=rollbacks,exit_code=status,checks=checks,
  execution_verified=verified,repair_execution_verified=verified and mode==1,native_material_law_unchanged=True,native_contact_kernels_unchanged=True,per_cycle_host_readback=False,frame_synchronization=True,physical_improvement_accepted=False,performance_accepted=False,platform_acceptance=False,
  scope='Exactly two free-node volumes, fixed topology and active body-list contents within each frame. Asynchronous finite checks reject at next frame or stop; arbitrary dynamic topology is unsupported.')
