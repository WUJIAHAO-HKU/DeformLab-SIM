"""Hold the platform lock and supervise a fresh cloth-pull recording."""
import argparse, fcntl, json, os, signal, subprocess, shutil
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
p=argparse.ArgumentParser();p.add_argument('--output-dir',required=True)
p.add_argument('--preview',action='store_true');p.add_argument('--native-display',action='store_true')
p.add_argument('--interior',action='store_true');p.add_argument('--render-samples',type=int,default=32)
p.add_argument('--physics-only',action='store_true')
a=p.parse_args();out=(ROOT/a.output_dir).resolve()
assert out.is_relative_to(ROOT/'results') and not out.exists()
lock=os.open(ROOT/'.cache/isaac/build-pipeline.lock',os.O_CREAT|os.O_RDWR|os.O_NOFOLLOW,0o600)
fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);out.mkdir(parents=True)
shutil.copy2(ROOT/'scripts/isaac_cloth_pull_video.py',out/'scene-source.py')
shutil.copy2(__file__,out/'runner-source.py')
if a.interior:shutil.copy2(ROOT/'deformable_tasks/interior.py',out/'interior-source.py')
if a.interior:shutil.copy2(ROOT/'deformable_tasks/specimen_appearance.py',out/'appearance-source.py')
env=os.environ.copy()
for key in ('PYTHONPATH','PYTHONHOME','PYTHONEXE','VIRTUAL_ENV','CONDA_PREFIX','LD_PRELOAD'):env.pop(key,None)
env.update(OMNI_KIT_ACCEPT_EULA='YES',OMNI_KIT_ALLOW_ROOT='1',XDG_CACHE_HOME=str(ROOT/'.cache/isaac'),TMPDIR=str(ROOT/'.cache/isaac/tmp'))
for key in ('HTTP_PROXY','HTTPS_PROXY','http_proxy','https_proxy'):env[key]='http://127.0.0.1:18767'
if a.native_display:
    s=json.loads((ROOT/'.cache/isaac-display/session.json').read_text());env.update(DISPLAY=s['display'],XAUTHORITY=s['xauthority_path'])
cmd=[str(ROOT/'upstream/IsaacSim/_build/linux-x86_64/release/python.sh'),str(ROOT/'scripts/isaac_cloth_pull_video.py'),'--output-dir',str(out),'--build-receipt',str(ROOT/'results/isaac-initial-runtime-egl-v4/receipt.json')]
if a.preview:cmd+=['--preview']
if a.native_display:cmd+=['--native-display']
if a.interior:cmd+=['--interior','--render-samples',str(a.render_samples)]
if a.physics_only:cmd+=['--physics-only']
record=dict(status='running',argv=cmd)
def save():(out/'runner.json').write_text(json.dumps(record,indent=2)+'\n')
def interrupt(sig,frame):raise KeyboardInterrupt
signal.signal(signal.SIGTERM,interrupt)
save()
try:
    with (out/'runtime.log').open('xb') as log:
        proc=subprocess.Popen(cmd,cwd=ROOT,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        record['pid']=proc.pid;save()
        try:code=proc.wait(timeout=2400)
        except (subprocess.TimeoutExpired,KeyboardInterrupt):
            os.killpg(proc.pid,signal.SIGTERM)
            try:proc.wait(timeout=20)
            except subprocess.TimeoutExpired:os.killpg(proc.pid,signal.SIGKILL);proc.wait(timeout=10)
            raise
    report=json.loads((out/'report.json').read_text())
    passed=code==0 and report.get('exit_code')==0 and report['status']=='pending_official_fast_exit' and all(c['passed'] for c in report['checks'].values())
    record.update(status='passed' if passed else 'failed',process_exit_code=code)
except BaseException as e:
    record.update(status='failed',error=repr(e));passed=False
finally:save();os.close(lock)
raise SystemExit(0 if passed else 1)
