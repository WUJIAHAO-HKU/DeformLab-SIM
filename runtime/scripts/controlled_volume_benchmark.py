"""Exploratory equal-cost stability comparison at a shared 960 Hz time step.

Both sides retain inversion failures but continue finite trajectories, so early
termination can never masquerade as a cheap complete baseline. NaN/Inf still
aborts and makes the pair ineligible. Invalid-horizon cost is not an accuracy
or useful-throughput measurement.
"""
import argparse,hashlib,json,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from deformable_tasks.comparison import compare
parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output-dir',default='results/volume-controlled-budget-v1')
parser.add_argument('--volume-sweeps',type=int,default=16)
args=parser.parse_args();out=(ROOT/args.output_dir).resolve()
if not out.is_relative_to(ROOT/'results'):raise ValueError('Project results only')
out.mkdir()
source={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in list((ROOT/'deformable_tasks').glob('*.py'))+[ROOT/'scripts/run_tasks.py',Path(__file__)]}
protocol=dict(source_sha256=source,hz=960,duration_s=4,candidate=dict(backend='volume-project',iterations=32,floor=.15,sweeps=args.volume_sweeps),
    baseline=dict(backend='native',initial_iterations=64),tasks=['multi_soft','soft_squeeze'],tolerance=.10,max_confirmation_attempts=2,
    selection='Candidate is frozen from stability pilots. Baseline iteration budget is selected and, if necessary, adjusted solely from full-horizon measured compute time.',
    failed_horizon='Both sides continue finite inverted trajectories while retaining permanent physical failure. Nonfinite states abort and invalidate a pair. Cost includes work after physical failure and does not measure useful throughput.',
    scope='First exploratory stability comparison. The candidate adds a volume admissibility domain, so identical E/nu/rho do not imply an identical constitutive solution. No force accuracy, general speedup, or statistical superiority claim.')
(out/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
report=dict(status='running',runs={},comparisons={},decisions=[],errors=[])
def save():(out/'campaign.json').write_text(json.dumps(report,indent=2)+'\n')
def run(label,backend,task,iters):
    for p,h in source.items():
        if hashlib.sha256((ROOT/p).read_bytes()).hexdigest()!=h:raise RuntimeError('Source changed: '+p)
    report['active_run']=label;save();folder=out/label
    command=[sys.executable,'scripts/tasks.py','run','--output-dir',str(folder),'--task',task,'--backend',backend,
        '--base-hz','960','--iterations',str(iters),'--observe-failed-horizon']
    if backend=='volume-project':command+=['--volume-floor','.15','--volume-sweeps',str(args.volume_sweeps)]
    with (out/(label+'.log')).open('w') as stream:code=subprocess.call(command,cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT)
    runner=json.loads((folder/'runner.json').read_text())
    record=json.loads((folder/'report.json').read_text()) if (folder/'report.json').exists() else {}
    load=json.loads((folder/'gpu-load.json').read_text())
    row=dict(exit_code=code,runner=runner,gpu_load=load,cases=record.get('cases',{}),errors=record.get('errors',[]))
    report['runs'][label]=row;save();print('DONE',label,flush=True)
    if (folder/task/'measurements.json').exists():
        with (out/(label+'-analysis.log')).open('w') as stream:subprocess.run([sys.executable,'scripts/analyze_tasks.py',str(folder)],cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT,check=True)
    return row
def usable(row,task):
    return row['runner'].get('process_exit_code')==0 and row['gpu_load']['observed_load_passed'] and task in row['cases'] and row['cases'][task]['completed_duration']
def backgrounds(row):
    g=row['gpu_load'];start=g['admission_window_start']
    return {(a['pid'],a['start_ticks']) for a in g['admission'][start]['apps']}
try:
    for task in protocol['tasks']:
        baseline=run(task+'-cal-native','native',task,64)
        candidate=run(task+'-cal-candidate','volume-project',task,32)
        if not (usable(baseline,task) and usable(candidate,task)):
            report['errors'].append(dict(task=task,reason='Full finite horizons / observed load / process gate failed; no valid budget calibration'));save();continue
        n=max(4,min(255,round(64*candidate['cases'][task]['timing']['control_physics_sync_s']/baseline['cases'][task]['timing']['control_physics_sync_s'])))
        for attempt in range(2):
            report['decisions'].append(dict(task=task,attempt=attempt,baseline_iterations=n,physics_scores_used=False));save()
            baseline=run(f'{task}-r{attempt}-native','native',task,n)
            candidate=run(f'{task}-r{attempt}-candidate','volume-project',task,32)
            if not (usable(baseline,task) and usable(candidate,task)):
                report['errors'].append(dict(task=task,attempt=attempt,reason='Invalid observed load or incomplete run'));save();break
            pair=compare(baseline['cases'][task],candidate['cases'][task],clean_gpu=backgrounds(baseline)==backgrounds(candidate))
            pair.pop('physical_accuracy_claim_eligible',None)
            pair.update(baseline_iterations=n,candidate_iterations=32,ground_truth_accuracy_established=False)
            report['comparisons'][f'{task}/{attempt}']=pair;save()
            if pair['budget_pair_eligible'] or not pair['gpu_observation_clean']:break
            n=max(4,min(255,round(n*pair['compute_time_ratio_candidate_over_control'])))
    report.update(status='completed',active_run=None);save()
except BaseException as exc:
    report.update(status='failed',error=repr(exc));save();raise
