"""Explicit diagnostic-only first geometric failure capture; no physics writes."""
import ctypes,json,os
from pathlib import Path
import numpy as np
from .cloth_stage_backend import sha
from .cloth_surface_reentry_backend import BACKEND
FLAG='DEFORMABLE_CLOTH_GEOMETRY_CHECKPOINT'

def requested(task,backend):
 value=os.environ.get(FLAG,'0')
 if value not in ('0','1'):raise ValueError('Geometry checkpoint must be explicit 0 or 1')
 if value=='0':return False
 if task not in ('cloth_grip','cloth_fold') or backend!=BACKEND or os.environ.get('DEFORMABLE_CLOTH_FULL_RATE','0')!='1':
  raise ValueError('Geometry checkpoint requires the reentry backend and full-rate cloth diagnostics')
 return True

def artifacts(root,base):
 root=Path(root);build=root/'results/cloth-geometry-checkpoint-backend-build-v1'
 m=json.loads((build/'build.json').read_text());proof=root/'results/cloth-geometry-checkpoint-binding-v1/summary.json';r=json.loads(proof.read_text())
 if m.get('status')!='compiled' or m.get('physical_ptx_byte_identical') is not True or m.get('read_only_geometry_checkpoint') is not True:raise ValueError('Read-only checkpoint build required')
 if sha(base['cloth_split_ptx'])!=m['ptx_sha256']:raise ValueError('Checkpoint changed physical PTX')
 if r.get('passed') is not True or len(r.get('checks',{}))!=36 or not all(r['checks'].values()):raise ValueError('Checkpoint CUDA guards did not pass')
 pins={**m['input_sha256'],**r['input_sha256'],str(build/'libcloth_split_backend.so'):m['library_sha256'],str(build/'contact.ptx'):m['ptx_sha256']}
 if any(sha(p)!=h for p,h in pins.items()):raise ValueError('Checkpoint build/proof changed')
 files=dict(base);files['geometry_checkpoint_base_manifest']=base['manifest'];files['geometry_checkpoint_selftest']=proof
 files.update(library=build/'libcloth_split_backend.so',manifest=build/'build.json',cloth_split_ptx=build/'contact.ptx')
 for name in ('cloth_intersection_reference.py','cloth_adjacent_intersections.py','cloth_geometry_reference.py'):files['geometry_reference_'+name]=root/'scripts'/name
 return files

class GeometryCheckpoint:
 def __init__(self,runtime,profile,dest):
  from cloth_intersection_reference import audit_intersections
  from cloth_adjacent_intersections import adjacent_pairs,audit_adjacent
  if len(runtime.bodies)!=1 or profile.cloth_split is None or not profile.cloth_split.failure_ring:raise ValueError('One cloth and active failure ring required')
  self.runtime=runtime;self.lib=profile.cloth_split.lib;self.dest=Path(dest)
  self.faces=np.array(runtime.bodies[0]['faces'],copy=True);self.pairs=adjacent_pairs(self.faces)
  self.audit=audit_intersections;self.adjacent=audit_adjacent;self.captured=None;self.frames=0;self.last_step=-1
  self.lib.cloth_history_geometry_checkpoint.argtypes=[ctypes.c_char_p,ctypes.c_ulonglong,ctypes.POINTER(ctypes.c_ulonglong)]
  self.path=self.dest/'geometry-checkpoint-frames.jsonl'
  if self.path.exists():raise ValueError('Fresh geometry observation required')
  self.capture(0)
 def capture(self,step):
  if step!=self.last_step+1:raise ValueError('Geometry observation must include every endpoint')
  points=self.runtime.bodies[0]['last']
  if not np.array_equal(self.faces,self.runtime.bodies[0]['faces']):raise ValueError('Geometry topology changed')
  a=self.audit(points,self.faces);b=self.adjacent(points,self.faces,self.pairs)
  valid=a['valid_nondegenerate_geometry'] and b['valid_nondegenerate_geometry']
  cross=a.get('proper_crossing_pairs',0)+b.get('proper_crossing_pairs',0)
  overlap=a.get('coplanar_overlap_pairs',0)+b.get('coplanar_overlap_pairs',0)
  report=dict(step=step,valid_nondegenerate_geometry=bool(valid),proper_crossing_pairs=cross,coplanar_overlap_pairs=overlap,nonadjacent=a,adjacent=b)
  failed=not valid or cross>0 or overlap>0
  with self.path.open('a') as f:f.write(json.dumps(report)+'\n')
  self.frames+=1;self.last_step=step
  if failed and self.captured is None:
   if step==0:raise RuntimeError('Initial geometry already invalid; no pre-step history exists')
   values=(ctypes.c_ulonglong*11)();target=self.dest/'first-geometry.ring.bin'
   rc=self.lib.cloth_history_geometry_checkpoint(str(target).encode(),step,values)
   if rc!=0:raise RuntimeError('First geometry checkpoint failed: '+str(rc))
   raw=np.fromfile(target,'<u4',count=16)
   if len(raw)!=16 or raw[0]!=0x46524831 or int(raw[8])!=values[0] or values[1]!=step:raise RuntimeError('Checkpoint frame/header mismatch')
   self.captured=dict(step=step,epoch=int(values[0]),report=list(values),file=str(target),sha256=sha(target),geometry=report,
    first_native_rejection_epoch=int(raw[9]),first_native_decode_epoch=int(raw[10]))
   np.savez_compressed(self.dest/'first-geometry.npz',nodes=points,faces=self.faces,step=step)
   (self.dest/'first-geometry.json').write_text(json.dumps(self.captured,indent=2)+'\n')
  if step%240==0 or (failed and self.captured and self.captured['step']==step):self.save()
 def save(self):
  result=dict(enabled=True,frames=self.frames,last_step=self.last_step,first_failure=self.captured,
   timing_eligible=False,physical_state_modified=False,platform_acceptance=False,
   scope='Every end-step adjacent/nonadjacent strict crossing, overlap and degeneracy. GPU ring is read-only pre-EE history; earlier native rejection may have frozen it. Not continuous collision detection.')
  (self.dest/'geometry-checkpoint.json').write_text(json.dumps(result,indent=2)+'\n')
  return result
