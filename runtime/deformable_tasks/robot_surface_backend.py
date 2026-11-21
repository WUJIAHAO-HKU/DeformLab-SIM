"""Named, experimental finite-table robot routes; selection is not acceptance."""
import hashlib,json
from pathlib import Path
ROUTES={
 'robot-surface-control':('cp-control','off'),
 'robot-surface-persistent':('cp-control','persistent'),
 'robot-mapping-surface-control':('volume-mapping','off'),
 'robot-mapping-surface-persistent':('volume-mapping','persistent'),
}
STREAM_ROUTES={name.replace('surface-','surface-stream-'):value for name,value in ROUTES.items()}
PARALLEL_ROUTES={name.replace('surface-','surface-parallel-'):value for name,value in ROUTES.items()}
ROUTES.update(STREAM_ROUTES)
ROUTES.update(PARALLEL_ROUTES)
BACKENDS=tuple(ROUTES)
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def _artifacts(root,foldername):
 root=Path(root);folder=root/'results'/foldername;build=folder/'build.json';planfile=folder/'plan.json';execution=folder/'execution.json';audit=folder/'independent-audit.json'
 b=json.loads(build.read_text());p=json.loads(planfile.read_text());s=json.loads(execution.read_text());a=json.loads(audit.read_text());lib=folder/'libphysx_robot_surface.so'
 if b['exit_code']!=0 or sha(planfile)!=b['plan_sha256'] or sha(lib)!=b['library_sha256']:raise ValueError('Frozen robot surface build changed')
 if s['status'] not in ('completed','completed_with_failures') or s.get('source_unchanged') is not True or len(s['cases'])!=4:raise ValueError('Complete original live campaign required')
 if any(sha(folder/name)!=digest for name,digest in p['source_sha256'].items()):raise ValueError('Frozen native adapter or task source changed')
 # Execution is a prerequisite for exposing an experimental route. Geometry,
 # material, work and cost regression failures remain visible in this receipt.
 for backend in ('cp-control','volume-mapping'):
  case=next(c for c in s['cases'] if c['backend']==backend and c['mode']=='persistent')
  key=backend+'-persistent'
  if not case.get('surface_execution_verified') or not a['cases'][key]['checks'].get('surface_actual_execution'):raise ValueError('Actual native surface execution not established')
 for path,digest in a['input_sha256'].items():
  if sha(path)!=digest:raise ValueError('Live trajectory audit changed: '+path)
 files=dict(library=lib,manifest=build,plan=planfile,execution=execution,audit=audit,entry=folder/'source/isaac_volume_grasp.py',bridge=folder/'source/cp_trial_bridge.py',surface_bridge=folder/'source/robot_surface_live_bridge.py',motion=folder/'source/grasp_motion.py',ptx=folder/'rows.ptx')
 return files
def artifacts(root):return _artifacts(root,'robot-surface-live-build-v2')
def stream_artifacts(root):
 files=_artifacts(root,'robot-surface-stream-live-build-v1')
 guards=Path(root)/'results/robot-surface-stream-guards-v1';summary=guards/'summary.json';protocol=guards/'protocol.json'
 s=json.loads(summary.read_text());p=json.loads(protocol.read_text())
 if not s['passed'] or sha(protocol)!=s['protocol_sha256'] or any(sha(k)!=v for k,v in p['input_sha256'].items()):raise ValueError('Stream native validation proof changed')
 files.update(guard_summary=summary,guard_protocol=protocol,cost_audit=files['plan'].parent/'cost-audit.json')
 return files
def parallel_artifacts(root):
 files=_artifacts(root,'robot-surface-parallel-live-build-v1')
 for key,name in (('runtime','robot-surface-parallel-runtime-v1'),('packing','robot-surface-parallel-rows-gpu-v2'),('guards','robot-surface-parallel-guards-v1'),('quarantine','robot-surface-parallel-quarantine-v1'),('rejections','robot-surface-parallel-rejections-v2')):
  folder=Path(root)/'results'/name;summary=folder/'summary.json';protocol=folder/'protocol.json'
  s=json.loads(summary.read_text());p=json.loads(protocol.read_text())
  if not s['passed'] or sha(protocol)!=s['protocol_sha256'] or any(sha(k)!=v for k,v in p['input_sha256'].items()):raise ValueError('Parallel surface proof changed: '+name)
  files[key+'_summary']=summary;files[key+'_protocol']=protocol
 files['cost_audit']=files['plan'].parent/'cost-audit.json'
 return files
def plan(config,backend,base_metadata,root):
 from .robot_config import default_robot_config
 if backend not in ROUTES:raise ValueError('Unknown robot surface route')
 if config!=default_robot_config():raise ValueError('This experimental surface route currently requires the frozen default robot task; no unverified configuration fallback')
 files=(parallel_artifacts if backend in PARALLEL_ROUTES else stream_artifacts if backend in STREAM_ROUTES else artifacts)(root);base,policy=ROUTES[backend];meta=dict(base_metadata)
 components=list(meta.get('selected_components',[]))
 components+=['finite_table_boundary_node_coverage','within_step_native_multiplier_history','original_row_native_copyback'] if policy=='persistent' else ['same_binary_surface_adapter_disabled']
 meta.update(parallel_contact_construction=backend in PARALLEL_ROUTES,immediate_fault_quarantine=backend in PARALLEL_ROUTES,stream_ordered_gpu_validation=backend in STREAM_ROUTES or backend in PARALLEL_ROUTES,name=backend,underlying_backend=base,surface_policy=policy,surface_artifacts={k:str(v) for k,v in files.items()},selected_components=components,execution_verified=False,latest_contact_candidate_selected=False,intrusive_synchronization=True,performance_acceptance=False,scope='Experimental articulated robot plus one volume and its authored finite horizontal table. Original native friction, articulation and material pipeline remain active. Live coverage was verified; volume-margin regression, work/material accuracy, lifecycle repeats and controlled cost remain open.')
 return meta
