from pathlib import Path
import ast,json,sys,types,tempfile,os
import numpy as np
from timing_surface_bridge import RobotSurfaceLive
root=Path(__file__).parent
old=ast.parse((root/'original_component_bridge.txt').read_text(encoding='utf-8'));new=ast.parse((root/'timing_surface_bridge.py').read_text(encoding='utf-8'))
def methods(t):return {x.name:ast.dump(x,include_attributes=False)for c in t.body if isinstance(c,ast.ClassDef)for x in c.body if isinstance(x,ast.FunctionDef)}
a,b=methods(old),methods(new);checks={n+'_unchanged':a[n]==b[n] for n in ['frame','snapshot','finish']};assert all(checks.values())
class Fn:
 def __init__(self,callback):self.callback=callback;self.calls=[]
 def __call__(self,*a):self.calls.append(a);return self.callback(*a)
class Lib:
 def __init__(self):
  self.values=[33,33,1,33,0,0,0,33,33,10,1,0,0,0,1,33]
  self.observer_robot_surface_start=Fn(lambda *a:0);self.observer_robot_surface_frame=Fn(lambda *a:0)
  def stats(out):
   for i,v in enumerate(self.values):out[i]=v
   return 0
  self.observer_robot_surface_stats=Fn(stats)
  def forbidden(*a):raise AssertionError('trace must never be requested')
  self.observer_robot_surface_component_trace=Fn(forbidden)
with tempfile.TemporaryDirectory()as d:
 path=Path(d)/'rows.ptx';path.write_text('mock',encoding='utf-8');lib=Lib();trial=types.SimpleNamespace(lib=lib,handle={'node_count':4},evidence={'hook_sha256':'fake'})
 os.environ.pop('DEFORMABLE_COMPONENT_TRACE',None)
 x=RobotSurfaceLive(trial,path,np.array([[0,1,2,3]],dtype=np.int32),[0,0,0],.12,32);x.frame(1);x.after_frame(1);assert x.finish(1)
 assert len(lib.observer_robot_surface_start.calls)==1 and len(lib.observer_robot_surface_frame.calls)==1 and not lib.observer_robot_surface_component_trace.calls
 checks['actual_class_init_no_trace_env_and_no_trace_call']=True
 for i in [0,1,2,3,4,7,8,10,12,13,14,15]:
  oldvalue=lib.values[i];lib.values[i]+=1
  try:x.after_frame(1)
  except RuntimeError:checks['counter_'+str(i)+'_rejected']=True
  else:raise AssertionError(i)
  lib.values[i]=oldvalue
 os.environ['DEFORMABLE_COMPONENT_TRACE']='forbidden'
 try:RobotSurfaceLive(trial,path,np.array([[0,1,2,3]]),[0,0,0],.12,32)
 except RuntimeError:checks['trace_env_rejected']=True
 else:raise AssertionError('env')
 os.environ.pop('DEFORMABLE_COMPONENT_TRACE')
print(json.dumps({'passed':all(checks.values()),'checks':checks},indent=2))
