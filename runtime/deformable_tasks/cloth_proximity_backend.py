"""Explicit geometry-triggered diagnostics; never a production fallback or cost sample."""
import ctypes as C,json,math,os
from pathlib import Path
from .cloth_stage_backend import ClothStages,sha
CONTROL='cloth-split-envelope-refresh16-compact-proximity-control'
BACKEND='cloth-split-envelope-refresh16-edge-query-graph-compact-proximity-contact'
BACKENDS=(CONTROL,BACKEND)
def configuration():
 p=Path(os.environ['DEFORMABLE_CLOTH_TRIGGER_CONFIG']);v=json.loads(p.read_text())
 if set(v)!={'nodes','threshold_m','span_steps','horizon_steps'}:raise ValueError('Exact explicit trigger configuration required')
 nodes=v['nodes'];t=v['threshold_m'];span=v['span_steps'];h=v['horizon_steps']
 if not isinstance(nodes,list) or len(nodes)!=6 or any(type(n) is not int or not 0<=n<100000 for n in nodes) or len(set(nodes))!=6:raise ValueError('Six distinct root nodes required')
 if type(t) not in (int,float) or not math.isfinite(t) or not 0<t<=.01 or type(span) is not int or not 1<=span<=64 or type(h) is not int or not 1<=h<=1000000:raise ValueError('Invalid trigger bounds')
 return p,v

def artifacts(root):
 from .compact_cloth_observer_backend import artifacts as base_artifacts
 files=base_artifacts(root);root=Path(root);build=root/'results/cloth-proximity-trigger-build-v1';m=json.loads((build/'build.json').read_text())
 if m.get('status')!='compiled' or not all(m.get(k) is True for k in ('pre_integration_trigger','contact_ptx_byte_identical','compact_dispatch_byte_identical','intrusive_synchronization')):raise ValueError('Verified geometry trigger required')
 if sha(files['cloth_split_ptx'])!=m['ptx_sha256']:raise ValueError('Physical PTX mismatch')
 files.update(library=build/'libcloth_split_backend.so',manifest=build/'build.json',cloth_split_ptx=build/'contact.ptx')
 if sha(files['library'])!=m['library_sha256'] or sha(files['cloth_split_ptx'])!=m['ptx_sha256']:raise ValueError('Trigger build changed')
 pins=dict(m['input_sha256'])
 for key,name,count in [('cloth_split_selftest','binding',1080),('cloth_stage_selftest','capture',58),('edge_query_selftest','query',195),('selftest','dispatch',10),('cache_selftest','cache',12),('cloth_trigger_selftest','trigger-verification',2530)]:
  p=root/('results/cloth-proximity-'+name+('-v2' if name=='trigger-verification' else '-v1')+'/summary.json');r=json.loads(p.read_text())
  if r.get('passed') is not True or len(r.get('checks',{}))!=count or not all(r['checks'].values()):raise ValueError('Trigger prerequisite failed: '+name)
  pins.update(r['input_sha256']);files[key]=p
 if any(sha(p)!=h for p,h in pins.items()):raise ValueError('Trigger prerequisite changed')
 p,v=configuration();files['cloth_trigger_config']=p
 return files

class ProximityStages(ClothStages):
 def __init__(self,lib,root,dest,extended_window=False):
  p,self.config=configuration();self.config_path=p;self.config_sha=sha(p)
  super().__init__(lib,root,dest,extended_window)
  try:
   lib.cloth_trigger_start.argtypes=[C.c_char_p,C.POINTER(C.c_uint),C.c_double,C.c_uint,C.c_uint]
   lib.cloth_trigger_report.argtypes=[C.POINTER(C.c_ulonglong),C.POINTER(C.c_double)]
   self.trigger_path=self.dest/'cloth-stages.trigger.jsonl'
   if os.path.lexists(self.trigger_path):raise ValueError('Trigger evidence exists')
   v=self.config
   if lib.cloth_trigger_start(str(self.prefix).encode(),(C.c_uint*6)(*v['nodes']),v['threshold_m'],v['span_steps'],v['horizon_steps']):raise RuntimeError('Cannot arm proximity trigger')
   self.trigger_path.symlink_to(Path(str(self.prefix)+'.trigger.jsonl'))
  except BaseException:
   lib.cloth_stage_stop((C.c_ulonglong*6)());self.stopped=True;raise
 def finish(self,steps):
  v=(C.c_ulonglong*8)();s=(C.c_double*3)();status=self.lib.cloth_trigger_report(v,s)
  fired=bool(v[2]);self.first=int(v[3]) if fired else 1;self.last=int(v[4]) if fired else steps
  e=super().finish(steps);rows=[];valid_rows=True
  with self.trigger_path.open() as f:
   for line in f:
    r=json.loads(line);i=len(rows)+1
    valid_rows &= r['step']==i and r['previous_end_step']==i-1 and type(r['triggered_now']) is bool and math.isfinite(r['distance_m']) and r['distance_m']>=0 and len(r['positions_m'])==18 and all(math.isfinite(x) for x in r['positions_m'])
    rows.append(r)
  actual_events=[r['step'] for r in rows if r['triggered_now']];eligible=[r['step'] for r in rows if r['distance_m']<=self.config['threshold_m']]
  expected_events=eligible[:1];expected_first=expected_events[0] if expected_events else 0
  expected_last=min(steps,expected_first+self.config['span_steps']-1) if expected_first else 0
  valid=status==0 and v[0]==1 and v[1]==steps==len(rows)==self.config['horizon_steps'] and v[7]==0 and e['exit_code']==0 and e['errors']==0 and e['steps']==steps and sha(self.config_path)==self.config_sha and valid_rows
  valid &= actual_events==expected_events and list(v)[2:7]==[int(bool(expected_events)),expected_first,expected_last,self.config['span_steps'],self.config['horizon_steps']]
  valid &= bool(rows) and s[0]==self.config['threshold_m'] and s[1]==min(r['distance_m'] for r in rows) and s[2]==rows[-1]['distance_m']
  if not fired:valid &= e['state_records']==e['kernel_pairs']==e['raw_bytes']==e.get('self_extra_bytes',0)==0
  else:
   with (self.dest/'cloth-stages.jsonl').open() as f:
    valid &= all(self.first<=json.loads(line)['step']<=self.last for line in f)
  e['files_sha256'][str(self.trigger_path)]=sha(self.trigger_path);e['storage_paths'][str(self.trigger_path)]=str(self.trigger_path.resolve())
  e.update(triggered=fired,trigger_monitor_valid=bool(valid),trigger_configuration=self.config,trigger_configuration_sha256=self.config_sha,trigger_samples=int(v[1]),trigger_minimum_distance_m=s[1],window=[int(v[3]),int(v[4])] if fired else None,preintegration_previous_endpoint_probe=True)
  e['capture_valid']=bool(e['capture_valid'] and valid and fired)
  e['diagnostic_valid']=bool(valid and (e['capture_valid'] if fired else True))
  return e
