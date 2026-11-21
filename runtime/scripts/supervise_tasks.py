"""Isolate Kit environment, enforce project lock, preserve actual process outcome."""
import argparse,fcntl,json,os,signal,subprocess,shutil,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
import sys
sys.path.insert(0,str(ROOT))
from deformable_tasks.gpu_load import LoadGuard,snapshot,query
from deformable_tasks.backends import CP_MODES,hook_library
p=argparse.ArgumentParser();p.add_argument('--output-dir',required=True);p.add_argument('args',nargs=argparse.REMAINDER)
a=p.parse_args();out=(ROOT/a.output_dir).resolve()
extra=a.args[1:] if a.args[:1]==['--'] else a.args
robot_config=None;robot_metadata=None
# Resolve the requested task before creating output or loading a GPU backend.
if '--task' in extra:requested_task=extra[extra.index('--task')+1]
else:requested_task=None
if '--config' in extra:
    requested_config=json.loads(Path(extra[extra.index('--config')+1]).read_text())
    if requested_task not in (None,'cloth_grip',requested_config.get('task')):raise ValueError('Task/config mismatch')
    requested_task=requested_config.get('task')
from deformable_tasks.weight_backend import WEIGHT_BACKENDS
requested_backend=extra[extra.index('--backend')+1] if '--backend' in extra else 'native'
from deformable_tasks.trajectory_recording import requested as recording_requested
full_rate_recording=recording_requested(requested_task,requested_backend,rendered='--render' in extra,suite='--suite' in extra,timing_only='--timing-only' in extra)
from deformable_tasks.kernel_profile import requested as profile_requested,artifacts as profile_artifacts
profile_mode=profile_requested(requested_task,requested_backend)
profile_files=profile_artifacts(ROOT,requested_backend) if profile_mode else {}
if profile_mode and ('--suite' in extra or '--render' in extra):raise ValueError('Kernel profile uses one unrendered task per process')
from deformable_tasks.robot_surface_backend import BACKENDS as ROBOT_SURFACE_BACKENDS
if requested_backend in ROBOT_SURFACE_BACKENDS and requested_task!='robot_volume':raise ValueError('Finite-table surface backend requires robot_volume; no task fallback')
if requested_backend in WEIGHT_BACKENDS and requested_task!='robot_volume':
    raise ValueError('Weight contact experiment requires the robot_volume route; no cloth/multibody fallback')
if requested_task=='robot_volume':
    from deformable_tasks.robot_config import default_robot_config,validate_robot_config,robot_plan,recorder_arguments,save_robot_config
    rp=argparse.ArgumentParser()
    rp.add_argument('--task');rp.add_argument('--config');rp.add_argument('--backend',default='native')
    rp.add_argument('--iterations',type=int);rp.add_argument('--base-hz',type=int);rp.add_argument('--duration',type=float)
    rp.add_argument('--render',action='store_true')
    ra=rp.parse_args(extra)
    robot_config=requested_config if ra.config else default_robot_config()
    if ra.iterations is not None:robot_config['solver']['position_iterations']=ra.iterations
    if ra.base_hz is not None:robot_config['solver']['hz']=ra.base_hz
    if ra.duration is not None:robot_config['duration_s']=ra.duration
    robot_config,robot_metadata=robot_plan(robot_config,ra.backend)

assert out.is_relative_to(ROOT/'results') and not out.exists()
lock=os.open(ROOT/'.cache/isaac/build-pipeline.lock',os.O_CREAT|os.O_RDWR|os.O_NOFOLLOW,0o600)
fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);out.mkdir(parents=True)
shutil.copytree(ROOT/'deformable_tasks',out/'source');shutil.copy2(ROOT/'scripts/run_tasks.py',out/'run_tasks.py')
shutil.copy2(__file__,out/'supervisor.py')
env=os.environ.copy()
for k in ('PYTHONPATH','PYTHONHOME','PYTHONEXE','VIRTUAL_ENV','CONDA_PREFIX','LD_PRELOAD'):env.pop(k,None)
env.update(OMNI_KIT_ACCEPT_EULA='YES',OMNI_KIT_ALLOW_ROOT='1',XDG_CACHE_HOME=str(ROOT/'.cache/isaac'),TMPDIR=str(ROOT/'.cache/isaac/tmp'))
# Native asset loading uses direct access, not the retired local proxy.
for k in ('HTTP_PROXY','HTTPS_PROXY','ALL_PROXY','http_proxy','https_proxy','all_proxy'):env.pop(k,None)
if robot_config is None and '--backend' in extra and extra[extra.index('--backend')+1] in CP_MODES:
    env['LD_PRELOAD']=str(hook_library(ROOT,extra[extra.index('--backend')+1]))
display=json.loads((ROOT/'.cache/isaac-display/session.json').read_text())
env.update(DISPLAY=display['display'],XAUTHORITY=display['xauthority_path'])
cmd=[str(ROOT/'upstream/IsaacSim/_build/linux-x86_64/release/python.sh'),str(ROOT/'scripts/run_tasks.py'),'--output-dir',str(out)]+extra
source_pins={}
if profile_mode:
    env['LD_PRELOAD']=str(profile_files['library'])
    env.update(OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1')
    inputs=[*profile_files.values(),*[ROOT/'scripts'/n for n in ('tasks.py','run_tasks.py','supervise_tasks.py','isaac_platform_smoke.py','isaac_deformable_baseline.py')],*list((ROOT/'deformable_tasks').glob('*.py'))]
    source_pins={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs}
if robot_config is not None:
    save_robot_config(out/'effective-task.json',robot_config)
    source_files=[ROOT/'scripts'/n for n in ('tasks.py','isaac_volume_grasp.py','cp_trial_bridge.py','grasp_motion.py','isaac_platform_smoke.py','isaac_deformable_baseline.py')]
    source_files += list((ROOT/'deformable_tasks').glob('*.py'))+[Path(__file__)]
    source_files += [Path(p) for p in robot_metadata.get('mapping_artifacts',{}).values()]
    source_files += [Path(p) for p in robot_metadata.get('surface_artifacts',{}).values()]
    for path in source_files:
        target=out/'pinned-source'/path.relative_to(ROOT);target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(path,target)
        source_pins[str(path)]=hashlib.sha256(path.read_bytes()).hexdigest()
    if robot_metadata['trial_mode']!='none':
        env['LD_PRELOAD']=str(ROOT/'work/kernel-observer/libphysx_cp_trial_hook.so')
        from deformable_tasks.weight_backend import WEIGHT_BACKENDS,artifacts
        if ra.backend in WEIGHT_BACKENDS:
            selected=artifacts(ROOT,ra.backend);env['LD_PRELOAD']=str(selected['library'])
            for path in [selected['ptx'],selected['manifest']]+([selected['query_ptx']] if 'query_ptx' in selected else []):source_pins[str(path)]=hashlib.sha256(path.read_bytes()).hexdigest()
        if 'surface_artifacts' in robot_metadata:env['LD_PRELOAD']=robot_metadata['surface_artifacts']['library']
        for path in [Path(env['LD_PRELOAD']),ROOT/'work/kernel-observer/physx_cp_conservative_trial.ptx']:
            source_pins[str(path)]=hashlib.sha256(path.read_bytes()).hexdigest()
    env.update(OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1')
    robot_entry=robot_metadata.get('surface_artifacts',{}).get('entry',str(ROOT/'scripts/isaac_volume_grasp.py'))
    cmd=[str(ROOT/'upstream/IsaacSim/_build/linux-x86_64/release/python.sh'),robot_entry,
         '--output-dir',str(out),'--build-receipt',str(ROOT/'results/isaac-initial-runtime-egl-v4/receipt.json'),'--native-display']
    cmd+=recorder_arguments(out/'effective-task.json',robot_config,ra.backend,ra.render)
record=dict(status='running',argv=cmd,load_mode='controlled',timing_eligible=False,full_rate_recording_requested=full_rate_recording)
if profile_mode:record.update(kernel_profile_requested=True,source_sha256=source_pins,performance_acceptance=False)
if robot_config is not None:record.update(task='robot_volume',backend=robot_metadata,source_sha256=source_pins,performance_acceptance=False)

def save():(out/'runner.json').write_text(json.dumps(record,indent=2)+'\n')
def interrupt(sig,frame):raise KeyboardInterrupt
signal.signal(signal.SIGTERM,interrupt);save()
try:
    guard=LoadGuard(out)
    policy_file=ROOT/'work/material-validation/resource-policy.json'
    shared_policy=None
    if policy_file.exists():
        requested=json.loads(policy_file.read_text())
        if requested.get('mode')!='shared-numeric-only':raise ValueError('Unknown resource policy')
        for directory in requested['result_roots']:
            boundary=(ROOT/directory).resolve()
            if not boundary.is_relative_to(ROOT/'results'):raise ValueError('Resource policy must be scoped to project results')
            if out.is_relative_to(boundary):shared_policy=requested
    if shared_policy:
        # This affects launch admission only. Never label shared runs as a
        # controlled performance experiment, even if a sample happens to be idle.
        free_mib=[float(v) for v in query(['--query-gpu=memory.free','--format=csv,noheader,nounits']).splitlines()]
        if len(free_mib)!=1 or free_mib[0]<6144:raise RuntimeError('Shared numerical mode requires at least 6144 MiB free GPU memory')
        observation=snapshot();guard.admission=[observation];guard.admission_window_start=None
        guard.baseline={app['pid']:app for app in observation['apps']}
        guard.policy.update(admission_kind='shared-numeric-only',minimum_initial_free_memory_mib=6144,
            scope='Shared GPU numerical validation; fixed simulated time and unchanged solver. All timing is ineligible for controlled performance claims.')
        guard.errors.append('Shared numerical mode: controlled timing admission intentionally not performed')
        guard.save();record.update(load_mode='shared-numeric-only',resource_policy=shared_policy,initial_free_gpu_memory_mib=free_mib[0]);save()
        (out/'resource-policy.json').write_text(json.dumps(shared_policy,indent=2)+'\n')
    else:guard.admit()
    with (out/'runtime.log').open('xb') as log:
        proc=subprocess.Popen(cmd,cwd=ROOT,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        record['pid']=proc.pid;save()
        guard.start(proc.pid)
        try:code=proc.wait(timeout=2400)
        except (subprocess.TimeoutExpired,KeyboardInterrupt):
            os.killpg(proc.pid,signal.SIGTERM)
            try:proc.wait(timeout=20)
            except subprocess.TimeoutExpired:os.killpg(proc.pid,signal.SIGKILL);proc.wait(timeout=10)
            raise
    record['process_exit_code']=code
    if not (out/'report.json').exists():raise RuntimeError('Native child exited without task report; exit code '+str(code)+'; inspect runtime.log')
    report=json.loads((out/'report.json').read_text())
    expected_status='pending_official_fast_exit' if robot_config is not None else 'completed_pending_process_exit'
    completed=code==0 and report.get('exit_code')==0 and report['status']==expected_status
    if robot_config is not None:
        unchanged=all(hashlib.sha256(Path(path).read_bytes()).hexdigest()==value for path,value in source_pins.items())
        record['source_unchanged']=unchanged
        completed=completed and unchanged
        physical_pass=completed and bool(report.get('checks')) and all(c.get('passed') is True for c in report['checks'].values())
        if 'surface_policy' in robot_metadata:
            surface=report.get('surface_runtime',{})
            binding=report.get('backend',{}).get('name')==robot_metadata['underlying_backend'] and surface.get('mode')==robot_metadata['surface_policy']
            execution=surface.get('execution_verified') is True if robot_metadata['surface_policy']=='persistent' else surface.get('mode')=='off'
            record.update(surface_binding_verified=bool(binding),surface_execution_verified=bool(execution))
            physical_pass=physical_pass and binding and execution
        record['robot_task_check_scope']='Full 7.5 s numerical task, not force accuracy or full lifecycle validation'
    else:
        if profile_mode:
            record['source_unchanged']=all(hashlib.sha256(Path(p).read_bytes()).hexdigest()==h for p,h in source_pins.items())
            completed=completed and record['source_unchanged']
        physical_pass=completed and all(c['status']=='passed' for c in report['cases'].values())
    record.update(status=('completed' if physical_pass else 'completed_with_regression_failures') if completed else 'failed',process_exit_code=code,
        all_physical_checks_passed=physical_pass)
except BaseException as e:record.update(status='failed',error=repr(e));completed=False
finally:
    if 'guard' in locals():
        guard.finish();record['gpu_load_observed_passed']=not guard.errors and bool(guard.samples)
        from deformable_tasks.cloth_split_backend import BACKENDS as CLOTH_SPLIT_BACKENDS
        record['timing_eligible']=bool(not (robot_config is not None and robot_metadata.get('intrusive_synchronization')) and not full_rate_recording and os.environ.get('DEFORMABLE_CLOTH_SPLIT_STAGES','0')=='0' and (not profile_mode or requested_backend in CLOTH_SPLIT_BACKENDS) and completed and record.get('all_physical_checks_passed') and record['load_mode']=='controlled' and record['gpu_load_observed_passed'])
    save();os.close(lock)
raise SystemExit(0 if completed and record.get('all_physical_checks_passed') else 1)
