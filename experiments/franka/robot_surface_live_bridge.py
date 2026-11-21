"""Explicit native launch-adapter lifecycle for the authored robot/table task."""
import ctypes as C,hashlib
from pathlib import Path
import numpy as np
class RobotSurfaceLive:
 def __init__(self,trial,ptx,tets,center,plane,iterations):
  self.lib=trial.lib;self.iterations=iterations;self.expected=iterations+1
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
  code=self.lib.observer_robot_surface_start(str(ptx).encode(),len(t),boundary.ctypes.data,len(boundary),center[0],center[1],.08,.08,plane,iterations)
  if code:raise RuntimeError('Surface adapter start rejected: '+str(code))
  self.evidence=dict(mode='persistent',row_ptx_sha256=hashlib.sha256(Path(ptx).read_bytes()).hexdigest(),hook_sha256=trial.evidence['hook_sha256'],boundary_nodes=len(boundary),frame_checks=0,progress=[],execution_verified=False,full_feedback_trajectory=True,intrusive_synchronization=True,performance_accepted=False,platform_acceptance=False)
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
  self.evidence['frame_checks']+=1
  if step%240==0:self.evidence['progress'].append(dict(step=step,statistics=v))
 def finish(self,steps):
  v=self.snapshot();passed=self.evidence['frame_checks']==steps and v[0]==v[1]==steps*self.expected and v[9]>0 and v[4]==v[12]==v[13]==0
  self.evidence.update(execution_verified=bool(passed),expected_steps=steps,expected_calls=steps*self.expected)
  return bool(passed)
