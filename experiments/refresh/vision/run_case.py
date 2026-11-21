"""Run one predeclared case, preserving all failures and unrelated GPU services.

No automatic cleanup/offload. Supervisor must verify the archived payload before
releasing scratch. A foreign compute process blocks strict timing before launch.
"""
from pathlib import Path
import argparse,fcntl,hashlib,json,os,shutil,signal,subprocess,sys,time
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[2]
sys.path.insert(0,str(HERE.parent/'budget'))
from gpu_monitor import snapshot
from heldout_protocol import verify,digest,write
from completion_contract import completion
LIB='6c321d72699984b12e60f0af9a4e09a8842a1e1a6bebab400ba393abdc00e5f6'
PTX={'B':'d91a9e77339ae4551256723764f3ad66218352fa18b1505466aab514d575ec39','D':'d91a9e77339ae4551256723764f3ad66218352fa18b1505466aab514d575ec39','dense':'eae120e2907826aec656eafa3bf7f877993075fd58d0cbb2b18c573625f3e9c4'}
def run(protocol_dir,case_name,output,timing_class):
    p=verify(protocol_dir);matches=[c for c in p['cases'] if c['name']==case_name];assert len(matches)==1
    cp=Path(protocol_dir)/matches[0]['case_file'];c=json.loads(cp.read_text());build=Path(c['component_build'])
    assert digest(build/'libphysx_robot_surface.so')==LIB
    assert digest(build/'rows.ptx')==PTX[c['arm']]
    if timing_class=='exploratory' and c['budget_status']=='frozen_from_controlled_calibration':raise ValueError('Controlled allocation cannot be run as exploratory confirmation')
    gpu=snapshot(os.getpid(),[])
    if not gpu['query_valid']:raise RuntimeError('GPU query invalid')
    if timing_class=='strict' and gpu['foreign_pids']:raise RuntimeError('Foreign GPU process: no strict launch; keep service intact')
    if min(float(g['free_memory'].split()[0]) for g in gpu['gpus'])<12288:raise RuntimeError('GPU free memory below12GiB')
    out=Path(output)
    if out.exists():raise FileExistsError('Do not overwrite a run')
    if shutil.disk_usage(out.parent).free<512*1024**2:raise RuntimeError('Less than512MiB free scratch')
    out.mkdir();write(out/'allowed-graphics.json',dict(allowed_graphics=[]))
    sys.path.insert(0,str(ROOT/'work/paper-generalization-v1/method'))
    import campaign_calibrated_v2 as common
    env=common.env(build);env.pop('DEFORMABLE_COMPONENT_TRACE',None)
    assert not env.get('DEFORMABLE_COMPONENT_TRACE')
    python=str(Path(env['ISAAC_PATH'])/'python.sh');auditpy=str(ROOT/'.venv/bin/python')
    argv=[python,str(HERE/'heldout_entry.py'),'--root',str(ROOT),'--case',str(cp),'--output',str(out)]
    rec=dict(case=case_name,source_protocol_sha256=digest(Path(protocol_dir)/'protocol.json'),case_sha256=digest(cp),timing_requested=timing_class,initial_gpu=gpu,argv=argv,status='starting',started_unix=time.time())
    write(out/'supervisor.json',rec)
    with (ROOT/'.cache/isaac/build-pipeline.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        monitor_cmd=[auditpy,str(HERE.parent/'budget/gpu_monitor.py'),'--root-pid',str(os.getpid()),'--out',str(out/'gpu-monitor.jsonl'),'--stop-file',str(out/'monitor.stop'),'--allowed-graphics',str(out/'allowed-graphics.json')]
        monitor=subprocess.Popen(monitor_cmd,cwd=ROOT,stdout=subprocess.DEVNULL,stderr=subprocess.PIPE,text=True)
        child=None
        try:
            with (out/'runtime.log').open('x') as log:
                child=subprocess.Popen(argv,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                rec.update(pid=child.pid,status='running');write(out/'supervisor.json',rec)
                try:code=child.wait(timeout=360)
                except subprocess.TimeoutExpired:
                    os.killpg(child.pid,signal.SIGTERM)
                    try:child.wait(timeout=15)
                    except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait()
                    code=124
        finally:
            if child is not None and child.poll() is None:
                os.killpg(child.pid,signal.SIGTERM)
                try:child.wait(timeout=15)
                except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait()
            (out/'monitor.stop').touch()
            try:_,err=monitor.communicate(timeout=15)
            except subprocess.TimeoutExpired:monitor.terminate();_,err=monitor.communicate(timeout=10)
            write(out/'gpu-monitor-lifecycle.json',dict(exit_code=monitor.returncode,allowed_graphics=[],stderr=err))
        rec['raw_exit_code']=code;rec['audits']={}
        native=json.loads((out/'process-receipt.json').read_text()) if (out/'process-receipt.json').exists() else {}
        rec['semantic_completion']=completion(native,code,c['config']);semantic_valid=rec['semantic_completion']['passed']
        jobs=[('physical','audit_side_push_v3.py','side-push-audit.json','execution_valid'),('backend','audit_side_push_backend.py','backend-execution-audit.json','record_valid'),('friction','audit_side_push_friction.py','friction-readback-audit.json','record_valid'),('image','audit_side_push_images.py','image-replay-audit.json','replay_integrity_passed')]
        if semantic_valid:
            for label,script,artifact,gate in jobs:
                with (out/(label+'-audit.log')).open('x') as log:
                    q=subprocess.run([auditpy,str(HERE/script),'--run',str(out),'--out',str(out/artifact)],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,timeout=180)
                rec['audits'][label]=dict(exit_code=q.returncode,passed=q.returncode==0 and (out/artifact).exists() and json.loads((out/artifact).read_text()).get(gate) is True)
            ray=ROOT/'work/paper-method-generalization-v1/reference/audit_side_push_rays.py'
            with (out/'ray-audit.log').open('x') as log:q=subprocess.run([auditpy,str(ray),'--run-dir',str(out),'--out',str(out/'ray-identity-audit.json')],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,timeout=180)
            rec['audits']['ray']=dict(exit_code=q.returncode,passed=q.returncode==0 and (out/'ray-identity-audit.json').exists() and json.loads((out/'ray-identity-audit.json').read_text()).get('record_valid') is True)
            with (out/'timing-audit.log').open('x') as log:q=subprocess.run([auditpy,str(HERE.parent/'budget/audit_timing.py'),'--run-dir',str(out),'--out',str(out/'timing-audit.json')],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,timeout=30)
            rec['timing_audit_exit_code']=q.returncode
        verify(protocol_dir)
        declared={str((HERE/n).resolve()):h for n,h in p['source_sha256'].items()}
        declared.update(p.get('external_source_sha256',{}))
        actual=json.loads((out/'actual-project-imports.json').read_text()) if (out/'actual-project-imports.json').exists() else {}
        rec['actual_imports_audit']=dict(passed=bool(actual) and all(declared.get(n)==h for n,h in actual.items()),module_count=len(actual),unfrozen_or_changed=[n for n,h in actual.items() if declared.get(n)!=h])
        rec['execution_valid']=semantic_valid and len(rec['audits'])==5 and all(v['passed'] for v in rec['audits'].values()) and rec['actual_imports_audit']['passed']
        rec.update(status='completed' if rec['execution_valid'] else 'retained_execution_failure',ended_unix=time.time(),payload_bytes=sum(f.stat().st_size for f in out.rglob('*') if f.is_file()))
        write(out/'supervisor.json',rec)
    print(json.dumps(rec));return 0 if rec['execution_valid'] else 1
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--protocol-dir',required=True);p.add_argument('--case-name',required=True);p.add_argument('--output',required=True);p.add_argument('--timing-class',choices=['strict','exploratory'],default='strict');a=p.parse_args()
    raise SystemExit(run(a.protocol_dir,a.case_name,a.output,a.timing_class))
