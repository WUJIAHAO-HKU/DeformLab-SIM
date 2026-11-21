"""Frozen first-round matrix; retain every failure and original measurement."""
import argparse,datetime,hashlib,json,os,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from deformable_tasks.config import TASKS,regression_configs
from deformable_tasks.comparison import compare,select_iterations

def main():
    p=argparse.ArgumentParser();p.add_argument('--output-dir',required=True);args=p.parse_args()
    out=(ROOT/args.output_dir).resolve()
    if not out.is_relative_to(ROOT/'results') or out.exists():raise ValueError('Fresh project results directory required')
    out.mkdir(parents=True)
    baseline_gpu=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,process_name,used_memory','--format=csv,noheader'],text=True).strip().splitlines()
    # Existing workbench is explicitly recorded; absence of NEW jobs is a weak
    # contamination screen, not exclusive GPU ownership or a utilization guarantee.
    baseline_pids={line.split(',')[0].strip() for line in baseline_gpu}
    sources={str(x.relative_to(ROOT)):hashlib.sha256(x.read_bytes()).hexdigest() for folder in ('deformable_tasks','scripts') for x in (ROOT/folder).glob('*.py') if folder=='deformable_tasks' or x.name in ('run_tasks.py','supervise_tasks.py','compare_task_backends.py','analyze_tasks.py','tasks.py')}
    manifest=dict(schema_version=2,created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        configs=regression_configs(),source_sha256=sources,git_head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        primary='Native 480Hz/64 iterations versus substep with 32 iterations and timing-selected frequency; same meshes/materials/controllers/thresholds.',
        secondary='Equal nominal work: native 480x64 versus substep 960x32. CP-control and CP-sequential single-volume only.',
        calibration=dict(candidate_iterations=32,effective_hz_bounds=[540,960],hz_quantum=60,max_attempts=3,
            selection='Full-rollout timing-only frequency correction. No threshold or physical score tuning.',
            cp='Separate 0.5-second timing fit at 1/32 iterations followed by full-horizon validation'),
        budget_tolerance=.10,full_duration_s=4.,repetitions=1,rendering_in_compute_budget=False,
        timing_scope='Synchronized CPU control + native physics; wall rollout including diagnostics also reported',
        existing_gpu_processes=baseline_gpu,limitation='Exploratory single repetition on shared GPU; no confidence intervals or force/real-world accuracy claims.',
        unsupported={'cp-sequential':['cloth_grip','cloth_fold','multi_soft']})
    (out/'preregistration.json').write_text(json.dumps(manifest,indent=2)+'\n')
    result=dict(status='running',runs={},comparisons={},calibration={},errors=[])
    def save():(out/'campaign.json').write_text(json.dumps(result,indent=2)+'\n')
    def run(label,backend,case_names=None,duration=None,iterations=None,render=False,base_hz=None):
        folder=out/label;extra=['--suite','--backend',backend]
        if case_names:extra+=['--cases',','.join(case_names)]
        if duration is not None:extra+=['--duration',str(duration),'--timing-only']
        if iterations is not None:extra+=['--iterations',str(iterations)]
        if base_hz is not None:extra+=['--base-hz',str(base_hz)]
        if render:extra+=['--render']
        cmd=[sys.executable,str(ROOT/'scripts/supervise_tasks.py'),'--output-dir',str(folder),'--']+extra
        result['active_run']=label;save();print('RUN '+label,flush=True)
        with (out/(label+'.log')).open('w') as log:code=subprocess.call(cmd,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
        report=json.loads((folder/'report.json').read_text()) if (folder/'report.json').exists() else {}
        runner=json.loads((folder/'runner.json').read_text()) if (folder/'runner.json').exists() else {}
        result['runs'][label]=dict(command=cmd,exit_code=code,runner=runner,cases=report.get('cases',{}),errors=report.get('errors',[]),pid=report.get('pid'))
        # Analyzer deliberately runs even when physics checks failed.
        if report.get('cases') and duration is None:
            with (out/(label+'-analysis.log')).open('w') as log:
                analysis_code=subprocess.call([sys.executable,str(ROOT/'scripts/analyze_tasks.py'),str(folder)],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
            result['runs'][label]['analysis_exit_code']=analysis_code
        save();return result['runs'][label]
    def clean(*runs):
        for run_data in runs:
            allowed=baseline_pids|{str(run_data['pid'])}
            for case in run_data['cases'].values():
                for sample in case['timing']['gpu_activity']:
                    if any(line.split(',')[0].strip() not in allowed for line in sample['processes']):return False
        return True
    save()
    try:
        native=run('native-regression','native')
        substep=run('substep-equal-work','substep')
        for name in regression_configs():
            if name in native['cases'] and name in substep['cases']:
                result['comparisons']['equal-work/'+name]=compare(native['cases'][name],substep['cases'][name],clean_gpu=clean(native,substep))
        save()
        # Short low-iteration fits were nonmonotonic in v1. Match measured
        # full-rollout time by bounded frequency correction, retaining failures.
        for name in TASKS:
            if not all(name in data['cases'] for data in (native,substep)):
                result['errors'].append('Missing calibration case '+name);save();continue
            control_time=native['cases'][name]['timing']['compute_s_per_sim_second']
            candidate_time=substep['cases'][name]['timing']['compute_s_per_sim_second']
            selected=max(540,min(960,round(960*control_time/candidate_time/60)*60))
            result['calibration'][name]=dict(control_s_per_sim_s=control_time,physical_scores_used=False,attempts=[])
            for attempt in range(1,4):
                result['calibration'][name]['attempts'].append(dict(attempt=attempt,effective_hz=selected,iterations=32));save()
                matched=run(f'substep-matched-{name}-{attempt}','substep',[name],iterations=32,base_hz=selected//2)
                if name not in matched['cases']:break
                pair=compare(native['cases'][name],matched['cases'][name],clean_gpu=clean(native,matched))
                result['comparisons'][f'matched-time/{name}/{attempt}']=pair;save()
                # Frequency tuning cannot repair external GPU interference.
                if pair['time_budget_within_tolerance'] or not pair['gpu_observation_clean']:break
                candidate_time=matched['cases'][name]['timing']['compute_s_per_sim_second']
                next_hz=max(540,min(960,round(selected*control_time/candidate_time/60)*60))
                if next_hz==selected:break
                selected=next_hz
        for name,c in regression_configs().items():
            if c['task']!='soft_squeeze':continue
            control=run('cp-control-'+name,'cp-control',[name]);candidate=run('cp-sequential-'+name,'cp-sequential',[name])
            if name in control['cases'] and name in candidate['cases']:
                result['comparisons']['cp-diagnostic/'+name]=compare(control['cases'][name],candidate['cases'][name],clean_gpu=clean(control,candidate))
                result['comparisons']['cp-diagnostic/'+name]['scope']='Same schedule, includes CP hook overhead; not assumed equal wall budget.'
            save()
        cp_cal=[]
        for label,backend,iters in [('cp-cal-control','cp-control',64),('cp-cal-low','cp-sequential',1),('cp-cal-high','cp-sequential',32)]:
            cp_cal.append(run(label,backend,['soft_squeeze'],.5,iters))
        if all('soft_squeeze' in data['cases'] for data in cp_cal):
            values=[data['cases']['soft_squeeze']['timing']['compute_s_per_sim_second'] for data in cp_cal]
            selected,reason=select_iterations(*values)
            result['calibration']['cp/soft_squeeze']=dict(target_low_high_s_per_sim_s=values,selected_iterations=selected,reason=reason,physical_scores_used=False)
            save()
            candidate=run('cp-matched-soft_squeeze','cp-sequential',['soft_squeeze'],iterations=selected)
            control=result['runs']['cp-control-soft_squeeze']
            if 'soft_squeeze' in control['cases'] and 'soft_squeeze' in candidate['cases']:
                pair=compare(control['cases']['soft_squeeze'],candidate['cases']['soft_squeeze'],clean_gpu=clean(control,candidate,*cp_cal))
                if not all(data['cases']['soft_squeeze']['completed_duration'] for data in cp_cal):pair.update(budget_pair_eligible=False,physical_accuracy_claim_eligible=False)
                result['comparisons']['cp-matched-time/soft_squeeze']=pair
            save()
        for backend in ('native','substep'):
            run('lifecycle-render-'+backend,backend,list(TASKS),.1,render=True)
        result.update(status='completed',active_run=None)
    except BaseException as exc:
        result.update(status='failed');result['errors'].append(repr(exc));raise
    finally:save()
    return 0

if __name__=='__main__':raise SystemExit(main())
