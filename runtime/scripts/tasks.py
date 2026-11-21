"""Unified user entry: native UI, task configurations, GPU runs and analysis."""
import argparse,json,os,subprocess,sys
from datetime import datetime,timezone
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from deformable_tasks.config import TASKS,LABELS,default_config,save_config
from deformable_tasks.backends import BACKENDS
from deformable_tasks.robot_config import default_robot_config,save_robot_config
ENTRY_TASKS=TASKS+('robot_volume',)

def main():
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='command',required=True)
    sub.add_parser('list');sub.add_parser('ui')
    sub.add_parser('repair-status',help='Repair selection and integration gaps; no GPU launch')
    reference=sub.add_parser('coupled-reference',help='CPU small-strain coupled contact reference; not a native GPU backend')
    reference.add_argument('--source-case',required=True);reference.add_argument('--output-dir',required=True)
    reference.add_argument('--hz',type=int,default=960);reference.add_argument('--duration',type=float,default=4.)
    reference.add_argument('--kelvin-voigt-s',type=float,default=0.)
    reference.add_argument('--impact-projection',action='store_true')
    reference.add_argument('--adaptive',action='store_true');reference.add_argument('--velocity-tolerance',type=float,default=2e-5)
    reference.add_argument('--position-tolerance',type=float,default=1e-8)
    reference.add_argument('--integrator',choices=('backward-euler','midpoint-hybrid','gauss-hybrid'),default='backward-euler')
    compare=sub.add_parser('compare');compare.add_argument('--output-dir',required=True)
    controlled=sub.add_parser('compare-contact');controlled.add_argument('--output-dir',required=True)
    controlled.add_argument('--backend',choices=('contact-adaptive','contact-local'),default='contact-local')
    controlled.add_argument('--iterations',type=int,default=48);controlled.add_argument('--repeats',type=int,default=3)
    volume_compare=sub.add_parser('compare-volume');volume_compare.add_argument('--output-dir',required=True)
    volume_compare.add_argument('--volume-sweeps',type=int,default=16)
    check_ui=sub.add_parser('check-ui');check_ui.add_argument('--output-dir',required=True);check_ui.add_argument('--scene',required=True)
    arch=sub.add_parser('architecture');arch.add_argument('--output-dir')
    movie=sub.add_parser('interior-video');movie.add_argument('--output-dir');movie.add_argument('--preview',action='store_true')
    movie.add_argument('--samples',type=int,choices=(16,32,64,128,256),default=64)
    init=sub.add_parser('init');init.add_argument('task',choices=ENTRY_TASKS);init.add_argument('file')
    for name in ('run','suite'):
        s=sub.add_parser(name);s.add_argument('--output-dir');s.add_argument('--render',action='store_true')
        s.add_argument('--profile',choices=('studio','pathtraced'),default=None)
        s.add_argument('--backend',choices=BACKENDS,default='native');s.add_argument('--iterations',type=int)
        s.add_argument('--duration',type=float);s.add_argument('--cases');s.add_argument('--timing-only',action='store_true')
        s.add_argument('--base-hz',type=int)
        s.add_argument('--volume-floor',type=float)
        s.add_argument('--volume-sweeps',type=int)
        s.add_argument('--material-damping-integrator',choices=('backward-euler','exponential'))
        s.add_argument('--audit-plan')
        s.add_argument('--observe-failed-horizon',action='store_true')
        if name=='run':s.add_argument('--task',choices=ENTRY_TASKS,default='cloth_grip');s.add_argument('--config')
    analyze=sub.add_parser('analyze');analyze.add_argument('directory')
    a=p.parse_args()
    if a.command=='repair-status':
        from deformable_tasks.repair_contract import integration_inventory
        print(json.dumps(integration_inventory(BACKENDS),indent=2));return 0
    if a.command=='coupled-reference':
        return subprocess.call([sys.executable,str(ROOT/'scripts/run_coupled_coupon_reference.py'),
            '--source-case',a.source_case,'--output-dir',a.output_dir,'--hz',str(a.hz),
            '--duration',str(a.duration),'--kelvin-voigt-s',str(a.kelvin_voigt_s),
            '--velocity-tolerance',str(a.velocity_tolerance),'--position-tolerance',str(a.position_tolerance),'--integrator',a.integrator]+
            (['--impact-projection'] if a.impact_projection else [])+(['--adaptive'] if a.adaptive else []),cwd=ROOT)
    if a.command=='compare':
        return subprocess.call([sys.executable,str(ROOT/'scripts/compare_task_backends.py'),'--output-dir',a.output_dir],cwd=ROOT)
    if a.command=='compare-contact':
        return subprocess.call([sys.executable,str(ROOT/'scripts/controlled_contact_benchmark.py'),'--output-dir',a.output_dir,
            '--backend',a.backend,'--iterations',str(a.iterations),'--repeats',str(a.repeats)],cwd=ROOT)
    if a.command=='compare-volume':
        return subprocess.call([sys.executable,str(ROOT/'scripts/controlled_volume_benchmark.py'),'--output-dir',a.output_dir,'--volume-sweeps',str(a.volume_sweeps)],cwd=ROOT)
    if a.command=='check-ui':
        return subprocess.call([sys.executable,str(ROOT/'scripts/supervise_task_workbench.py'),'--probe','tasks','--receipt','results/isaac-initial-runtime-egl-v4/receipt.json','--scene',a.scene,'--output-dir',a.output_dir],cwd=ROOT)
    if a.command=='list':
        for task,label in zip(ENTRY_TASKS,LABELS+('Franka volume grip / squeeze / lift / release',)):print(task+' : '+label)
        return 0
    if a.command=='init':
        if a.task=='robot_volume':save_robot_config(a.file,default_robot_config())
        else:save_config(a.file,default_config(a.task))
        print(Path(a.file).resolve());return 0
    if a.command=='architecture':
        output=a.output_dir or ('results/architecture-'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S'))
        return subprocess.call([sys.executable,str(ROOT/'scripts/supervise_architecture.py'),'--output-dir',output],cwd=ROOT)
    if a.command=='interior-video':
        output=a.output_dir or ('results/interior-cloth-'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S'))
        cmd=[sys.executable,str(ROOT/'scripts/supervise_cloth_pull.py'),'--interior','--native-display','--render-samples',str(a.samples),'--output-dir',output]
        if a.preview:cmd.append('--preview')
        code=subprocess.call(cmd,cwd=ROOT)
        if code:return code
        code=subprocess.call([sys.executable,str(ROOT/'scripts/verify_interior_video.py'),str(ROOT/output)],cwd=ROOT)
        if code:return code
        return subprocess.call([sys.executable,str(ROOT/'scripts/build_cloth_replay.py'),str(ROOT/output)],cwd=ROOT)
    if a.command in ('run','suite'):
        out=a.output_dir or ('results/task-'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S.%f'))
        cmd=[sys.executable,str(ROOT/'scripts/supervise_tasks.py'),'--output-dir',out,'--']
        if a.command=='suite':cmd+=['--suite']
        else:
            cmd+=['--task',a.task]
            if a.config:cmd+=['--config',str(Path(a.config).resolve())]
        if a.render:cmd+=['--render']
        if a.profile:cmd+=['--profile',a.profile]
        cmd+=['--backend',a.backend]
        if a.iterations is not None:cmd+=['--iterations',str(a.iterations)]
        if a.duration is not None:cmd+=['--duration',str(a.duration)]
        if a.base_hz is not None:cmd+=['--base-hz',str(a.base_hz)]
        if a.volume_floor is not None:cmd+=['--volume-floor',str(a.volume_floor)]
        if a.volume_sweeps is not None:cmd+=['--volume-sweeps',str(a.volume_sweeps)]
        if a.material_damping_integrator is not None:cmd+=['--material-damping-integrator',a.material_damping_integrator]
        if a.audit_plan:cmd+=['--audit-plan',str(Path(a.audit_plan).resolve())]
        if a.observe_failed_horizon:cmd+=['--observe-failed-horizon']
        if a.cases:cmd+=['--cases',a.cases]
        if a.timing_only:cmd+=['--timing-only']
        code=subprocess.call(cmd,cwd=ROOT)
        if a.command=='suite' and (ROOT/out/'report.json').exists():
            subprocess.call([sys.executable,str(ROOT/'scripts/analyze_tasks.py'),str(ROOT/out)],cwd=ROOT)
        return code
    if a.command=='analyze':return subprocess.call([sys.executable,str(ROOT/'scripts/analyze_tasks.py'),a.directory],cwd=ROOT)
    import fcntl
    lock=os.open(ROOT/'.cache/isaac/build-pipeline.lock',os.O_CREAT|os.O_RDWR|os.O_NOFOLLOW,0o600)
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    display=json.loads((ROOT/'.cache/isaac-display/session.json').read_text())
    env=os.environ.copy()
    for k in ('PYTHONPATH','PYTHONHOME','PYTHONEXE','VIRTUAL_ENV','CONDA_PREFIX','LD_PRELOAD'):env.pop(k,None)
    env.update(DISPLAY=display['display'],XAUTHORITY=display['xauthority_path'])
    try:return subprocess.call(['bash',str(ROOT/'scripts/launch_deformable_lab.sh')],cwd=ROOT,env=env,pass_fds=(lock,))
    finally:os.close(lock)

if __name__=='__main__':raise SystemExit(main())
