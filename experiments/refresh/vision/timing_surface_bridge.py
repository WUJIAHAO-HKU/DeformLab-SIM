"""Explicit native launch-adapter lifecycle for the authored robot/table task."""
import ctypes as C,hashlib,os
from pathlib import Path
import numpy as np
class RobotSurfaceLive:
 def __init__(self,trial,ptx,tets,center,plane,iterations,reference):
  self.lib=trial.lib;self.iterations=iterations;self.expected=iterations+1
  import json
  meta=json.loads((Path(ptx).parent/'material-reference.json').read_text())
  data=Path(ptx).parent/'material-reference.npz'
  if hashlib.sha256(data.read_bytes()).hexdigest()!=meta['reference_sha256']:raise RuntimeError('Material-reference bytes changed')
  with np.load(data,allow_pickle=False)as ref:
   assert np.allclose(np.asarray(reference['rest'],dtype=np.float32)-np.asarray(reference['rest'],dtype=np.float32).mean(0),ref['rest']-ref['rest'].mean(0),rtol=0,atol=2e-7)
   assert np.array_equal(np.asarray(tets,dtype=np.uint32),ref['elements'])
   assert np.allclose(reference['mass'],ref['mass'],rtol=1e-6,atol=1e-11) and np.isclose(float(np.sum(reference['mass'])),float(np.sum(ref['mass'])),rtol=2e-6,atol=1e-10)
   assert reference['young_pa']==float(ref['young_pa'])
   assert reference['poisson']==float(ref['poisson'])
   assert reference['hz']==float(ref['hz'])==960 and iterations==24
  self.reference_metadata=meta
  if os.environ.get('DEFORMABLE_COMPONENT_TRACE'):raise RuntimeError('Timing bridge requires no full-round trace')
  t=np.asarray(tets)
  if t.ndim!=2 or t.shape[1]!=4 or not np.issubdtype(t.dtype,np.integer) or np.any(t<0) or int(t.max())>=trial.handle['node_count']:raise ValueError('Invalid native topology')
  from collections import Counter
  faces=Counter(tuple(sorted(int(tet[i]) for i in face)) for tet in t for face in ((0,1,2),(0,1,3),(0,2,3),(1,2,3)))
  if any(n>2 for n in faces.values()):raise ValueError('Nonmanifold volume boundary')
  nodes=sorted({v for face,n in faces.items() if n==1 for v in face});rows=[]
  for node in nodes:
   ti=int(np.flatnonzero(np.any(t==node,axis=1))[0]);corner=int(np.flatnonzero(t[ti]==node)[0]);rows.append([node,ti,corner,0])
  boundary=np.asarray(rows,dtype=np.uint32)
  self.lib.observer_robot_surface_start.argtypes=[C.c_char_p,C.c_uint,C.c_void_p,C.c_uint,*[C.c_double]*5,C.c_uint];self.lib.observer_robot_surface_start.restype=C.c_int
  self.lib.observer_robot_surface_frame.argtypes=[C.c_uint];self.lib.observer_robot_surface_frame.restype=C.c_int
  self.lib.observer_robot_surface_stats.argtypes=[C.POINTER(C.c_ulonglong)];self.lib.observer_robot_surface_stats.restype=C.c_int
  self.lib.observer_robot_surface_component_trace.argtypes=[C.POINTER(C.c_ulonglong),C.c_uint]
  self.lib.observer_robot_surface_component_trace.restype=C.c_int
  self.trace_enabled=os.environ.get('DEFORMABLE_DISABLE_EVENT_TRACE')!='1'
  self.event_path=Path(os.environ['DEFORMABLE_EVENT_TRACE']) if self.trace_enabled else None
  if self.trace_enabled and self.event_path.exists():raise RuntimeError('Fresh event trace required')
  code=self.lib.observer_robot_surface_start(str(ptx).encode(),len(t),boundary.ctypes.data,len(boundary),center[0],center[1],.08,.08,plane,iterations)
  if code:raise RuntimeError('Surface adapter start rejected: '+str(code))
  self.evidence=dict(material_reference=self.reference_metadata,full_round_trace_enabled=self.trace_enabled,trace_scope='Development event indicators read back once per physical step; timing includes this overhead',mode='persistent',row_ptx_sha256=hashlib.sha256(Path(ptx).read_bytes()).hexdigest(),hook_sha256=trial.evidence['hook_sha256'],boundary_nodes=len(boundary),frame_checks=0,progress=[],execution_verified=False,full_feedback_trajectory=True,intrusive_synchronization=True,performance_accepted=False,platform_acceptance=False)
 def frame(self,step):
  code=self.lib.observer_robot_surface_frame(step)
  if code:raise RuntimeError('Surface frame rejected: '+str(code))
 def snapshot(self):
  values=(C.c_ulonglong*16)();code=self.lib.observer_robot_surface_stats(values)
  if code:raise RuntimeError('Surface statistics unavailable: '+str(code))
  values=list(values);self.evidence['statistics']=values;return values
 def after_frame(self,step):
  v=self.snapshot();valid=v[0]==v[1]==step*self.expected and v[2]==step and v[3]==self.expected and v[4]==v[12]==v[13]==0 and v[7]==v[8]==v[0] and v[10]==step and v[14]==step and v[15]==self.expected
  if not valid:raise RuntimeError('Incomplete surface execution: '+repr(v))
  if self.trace_enabled:
   raw=(C.c_ulonglong*(24*self.expected))()
   if self.lib.observer_robot_surface_component_trace(raw,self.expected):raise RuntimeError('Event trace unavailable')
   rows=np.ctypeslib.as_array(raw).reshape(self.expected,24)
   if not(np.all(rows[:,0]==step) and np.array_equal(rows[:,1],np.arange(self.expected)) and np.all(rows[:,23]==0x45565433)):raise RuntimeError('Event trace identity failed')
   with self.event_path.open('ab')as f:f.write(rows.astype('<u8').tobytes())
  self.evidence['frame_checks']+=1
  if step%240==0:self.evidence['progress'].append(dict(step=step,statistics=v))
 def finish(self,steps):
  v=self.snapshot();passed=self.evidence['frame_checks']==steps and v[0]==v[1]==steps*self.expected and v[9]>0 and v[4]==v[12]==v[13]==0
  raw=(C.c_ulonglong*(24*self.expected))()
  if self.lib.observer_robot_surface_component_trace(raw,self.expected):raise RuntimeError('Cumulative counters unavailable')
  totals=np.ctypeslib.as_array(raw).reshape(self.expected,24)[-1]
  if int(totals[23])!=0x45565433 or int(totals[0])!=steps or int(totals[17])!=steps*self.expected:raise RuntimeError('Cumulative counter identity failed')
  self.evidence['contact_refresh_counters']=dict(inner_rounds=int(totals[17]),refreshes=int(totals[18]),pre_final_triggers=int(totals[19]),material_triggers=int(totals[20]),refresh_fraction=float(totals[18]/totals[17]))
  self.evidence.update(execution_verified=bool(passed),expected_steps=steps,expected_calls=steps*self.expected)
  return bool(passed)
