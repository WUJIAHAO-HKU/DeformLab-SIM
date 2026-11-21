"""Diagnostic root geometry, kept separate from repair and performance acceptance."""
import ctypes,json,math
from fractions import Fraction
from pathlib import Path
import numpy as np
from .cone_backend import sha
BACKENDS={'volume-batched-geometry-control':0,'volume-batched-geometry-mapping':1}
DT=np.dtype([('calls','<u8'),('tet_checks','<u8'),('invalid_tets','<u8'),('nonfinite_tets','<u8'),('first_step','<u8'),('first_cycle','<u8'),('minimum_j','<f8'),('critical_tet','<u4'),('padding','<u4'),('before','<f4',(4,4)),('after','<f4',(4,4))])
STAGES=('pre_integrate','cp_partition','native_average','mapped_average','apply_external','step','finalize','other_sb')
NAMES=('sb_gm_preIntegrateLaunch','sb_gm_cp_solveTetrahedronsPartitionLaunch','sb_gm_cp_averageVertsLaunch',None,'sb_gm_applyExternalDeltasLaunch','sb_gm_stepSoftbodyLaunch','sb_gm_finalizeVelocitiesLaunch')
def extend_artifacts(root,files):
 root=Path(root);build=root/'results/batched-cp-geometry-binding-build-v1'
 files.update(library=build/'libbatched_cp_geometry.so',manifest=build/'build.json',geometry_ptx=root/'results/batched-cp-geometry-build-v1/geometry.ptx',geometry_primitive=root/'results/batched-cp-geometry-owned-v1/summary.json',geometry_binding=root/'results/batched-cp-geometry-binding-owned-v1/summary.json',geometry_streams=root/'results/batched-cp-geometry-streams-v1/summary.json')
 for key in ('manifest','geometry_primitive','geometry_binding','geometry_streams'):
  r=json.loads(files[key].read_text())
  if key=='manifest':
   if r['status']!='compiled' or sha(files['library'])!=r['library_sha256'] or Path(r['geometry_ptx'])!=files['geometry_ptx']:raise ValueError('Changed geometry binding')
  elif r.get('passed') is not True:raise ValueError('Geometry prerequisite failed')
  for p,h in r['input_sha256'].items():
   if sha(p)!=h:raise ValueError('Changed geometry prerequisite '+p)
 return files
class GeometryObserver:
 def __init__(self,lib,files,prefix):
  self.lib=lib;self.prefix=Path(prefix);self.stopped=False
  lib.batched_geometry_start.argtypes=[ctypes.c_char_p,ctypes.c_char_p];lib.batched_geometry_stop.argtypes=[ctypes.POINTER(ctypes.c_ulonglong)]
  if lib.batched_geometry_start(str(files['geometry_ptx']).encode(),str(self.prefix).encode()):raise RuntimeError('Cannot start root geometry observer')
 def stop(self):
  if self.stopped:raise RuntimeError('Geometry observer stopped twice')
  values=(ctypes.c_ulonglong*4)();status=self.lib.batched_geometry_stop(values);self.stopped=True
  return self.prefix,list(values),status

def exact_det(q):
 a=[[Fraction(float(q[i,j]))-Fraction(float(q[0,j])) for j in range(3)] for i in (1,2,3)];u,v,w=a
 return u[0]*(v[1]*w[2]-v[2]*w[1])+u[1]*(v[2]*w[0]-v[0]*w[2])+u[2]*(v[0]*w[1]-v[1]*w[0])
def evidence(prefix,values,status,native):
 prefix=Path(prefix);meta_path=Path(str(prefix)+'.json');blob=Path(str(prefix)+'.bin');ref_path=Path(str(prefix)+'-reference.bin');m=json.loads(meta_path.read_text())
 calls,errors,bodies,scratch=map(int,values);expected=[int(native.get(n,0)) if n else int(native.get(NAMES[2],0)) for n in NAMES]
 expected.append(sum(v for k,v in native.items() if k.startswith('sb_') and k not in NAMES))
 checks=dict(clean=status==errors==m.get('errors')==0,ready=m.get('ready') is True,host_coverage=m.get('native_stage_calls')==expected and sum(expected)==calls and calls>0,body_count=bodies==len(m.get('bodies',[]))==2,statistics_ABI=m.get('statistics_bytes')==DT.itemsize==192 and m.get('stages')==8)
 records=[];positive=True;published_positive=True
 if checks['ready']:
  raw=np.fromfile(blob,dtype=DT);reference=np.fromfile(ref_path,dtype='<f8')
  if raw.size!=16 or reference.size!=sum(b['tets'] for b in m['bodies']):raise ValueError('Incomplete geometry evidence')
  checks['positive_references']=bool(np.isfinite(reference).all() and (reference>0).all());offset=0
  for body,b in enumerate(m['bodies']):
   for stage,label in enumerate(STAGES):
    r=raw[body*8+stage];n=int(r['calls']);bad=int(r['invalid_tets']);nonfinite=int(r['nonfinite_tets']);minj=float(r['minimum_j']) if n and math.isfinite(float(r['minimum_j'])) else None
    checks[f'{body}/{label}/coverage']=n==expected[stage] and int(r['tet_checks'])==n*b['tets']
    feasible=not n or (bad==nonfinite==0 and minj is not None and minj>0);positive &= feasible
    if stage!=2:published_positive &= feasible
    item=dict(body=body,stage=label,calls=n,tet_checks=int(r['tet_checks']),invalid_tets=bad,nonfinite_tets=nonfinite,minimum_j=minj,positive=feasible)
    if bad or nonfinite:
     tet=int(r['critical_tet']);step=int(r['first_step']);cycle=int(r['first_cycle']);witness_ok=0<=tet<b['tets'] and 0<step<=native.get(NAMES[0],0) and cycle<=native.get(NAMES[2],0)+1
     finite_before=bool(np.isfinite(r['before'][:,:3]).all());finite_after=bool(np.isfinite(r['after'][:,:3]).all())
     before=exact_det(r['before'][:,:3]) if finite_before else None;after=exact_det(r['after'][:,:3]) if finite_after else None
     checks[f'{body}/{label}/witness']=witness_ok and ((after is not None and after<=0) or not finite_after)
     item['first_witness']=dict(step=step,cycle=cycle,tet=tet,reference_determinant=float(reference[offset+tet]) if witness_ok else None,exact_before=str(before) if before is not None else None,exact_after=str(after) if after is not None else None,new_inversion=before is not None and before>0 and after is not None and after<=0)
    else:checks[f'{body}/{label}/no_spurious_witness']=int(r['first_step'])==0
    records.append(item)
   offset+=b['tets']
 else:positive=published_positive=False
 paths=[p for p in (meta_path,blob,ref_path) if p.exists()]
 return dict(capture_valid=all(checks.values()),checks=checks,stage_records=records,all_observed_root_states_positive=bool(positive),excluding_native_average_root_states_positive=bool(published_positive),native_average_exclusion_is_not_acceptance=True,calls=calls,errors=errors,bodies=bodies,scratch_bytes=scratch,exit_code=status,input_sha256={str(p):sha(p) for p in paths},physical_buffers_modified=False,performance_accepted=False,platform_acceptance=False,scope='All intercepted sb_ launch endpoints and separate post-mapping average; root tetrahedra only. Native average before correction is reported separately. CP replicas and continuous trajectories remain unobserved.')
