"""Alternating repeated full-horizon pairs under observed low GPU load.

Only measured compute time selects iterations. Never infer equal cost from
iteration products or accept early termination as a cheap successful rollout.
"""
import argparse,hashlib,json,statistics,subprocess,sys
from datetime import datetime,timezone
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from deformable_tasks.comparison import compare

def main():
    p=argparse.ArgumentParser();p.add_argument('--output-dir',required=True)
    p.add_argument('--backend',choices=('contact-adaptive','contact-local'),default='contact-adaptive')
    p.add_argument('--iterations',type=int,default=48);p.add_argument('--repeats',type=int,default=3)
    a=p.parse_args();out=(ROOT/a.output_dir).resolve()
    if not out.is_relative_to(ROOT/'results'):raise ValueError('Project results only')
    out.mkdir()
    paths=list((ROOT/'deformable_tasks').glob('*.py'))+[Path(__file__),ROOT/'scripts/run_tasks.py',ROOT/'scripts/supervise_tasks.py',ROOT/'scripts/analyze_tasks.py']
    source={str(path.relative_to(ROOT)):hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}
    protocol=dict(created_utc=datetime.now(timezone.utc).isoformat(),source_sha256=source,
        tasks=['cloth_grip','cloth_fold'],backend=a.backend,initial_candidate_iterations=a.iterations,
        baseline=dict(hz=480,iterations=64),repeats=a.repeats,tolerance=.10,
        selection='One calibration pair, then one shared iteration adjustment using the median measured task cost ratio only, frozen before confirmation. No physics-score tuning within confirmation.',
        order='Even repeat: baseline then candidate. Odd repeat: candidate then baseline.',
        load='Each process needs ten consecutive idle observations and continuous process, pmon, memory, clock and temperature audit. Across each pair require identical background process identities. This is not exclusive-device reservation.',
        eligibility='Full horizons, same physical input, actual synchronized compute time within 10%, source hashes unchanged, clean process exits and observed low background load. Physics and task quality reported separately.',
        accuracy='No ground-truth force/deformation reference; report penetration/task feasibility only, not general accuracy superiority.')
    (out/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
    report=dict(status='running',runs={},selected_iterations={},pairs=[],summary={})
    def save():(out/'campaign.json').write_text(json.dumps(report,indent=2)+'\n')
    def run(label,backend,cases,iters):
        for path,h in source.items():
            if hashlib.sha256((ROOT/path).read_bytes()).hexdigest()!=h:raise RuntimeError('Frozen source changed: '+path)
        folder=out/label;report['active_run']=label;save()
        with (out/(label+'.log')).open('w') as stream:
            code=subprocess.call([sys.executable,'scripts/supervise_tasks.py','--output-dir',str(folder),'--',
                '--suite','--cases',','.join(cases),'--backend',backend,'--iterations',str(iters)],cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT)
        runner=json.loads((folder/'runner.json').read_text())
        if runner.get('process_exit_code')!=0:raise RuntimeError('Incomplete process: '+str(folder))
        result=json.loads((folder/'report.json').read_text());guard=json.loads((folder/'gpu-load.json').read_text())
        with (out/(label+'-analysis.log')).open('w') as stream:
            subprocess.run([sys.executable,'scripts/analyze_tasks.py',str(folder)],cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT,check=True)
        analysis=json.loads((folder/'interaction-analysis.json').read_text())
        row=dict(exit_code=code,runner=runner,cases=result['cases'],analysis=analysis,gpu_load=guard)
        report['runs'][label]=row;save()
        if not guard['observed_load_passed']:raise RuntimeError('Background GPU load invalidated run: '+label)
        print('DONE',label,flush=True);return row
    def identity(row):
        g=row['gpu_load'];index=g['admission_window_start']
        return sorted({(v['pid'],v['start_ticks']) for v in g['admission'][index]['apps']})
    def pair(control,candidate,task,repeat):
        result=compare(control['cases'][task],candidate['cases'][task],clean_gpu=identity(control)==identity(candidate))
        # The existing helper's all-checks gate is an execution check; it cannot
        # certify physical accuracy without an independent reference.
        result.pop('physical_accuracy_claim_eligible',None)
        result.update(task=task,repeat=repeat,ground_truth_accuracy_established=False)
        for name,row in [('baseline',control),('candidate',candidate)]:
            c=row['cases'][task];result[name]=dict(maximum_bench_penetration_mm=1000*c['maximum_bench_penetration_m'],
                minimum_volume_ratio=c['minimum_volume_ratio'],compute_s=c['timing']['control_physics_sync_s'],
                physical_checks_passed=all(c['checks'].values()),interaction=row['analysis']['cases'][task])
        return result
    try:
        control=run('calibration-native','native',protocol['tasks'],64)
        candidate=run('calibration-candidate',a.backend,protocol['tasks'],a.iterations)
        ratios=[control['cases'][task]['timing']['control_physics_sync_s']/candidate['cases'][task]['timing']['control_physics_sync_s'] for task in protocol['tasks']]
        n=max(4,min(128,round(a.iterations*statistics.median(ratios))))
        report['selected_iterations']={task:n for task in protocol['tasks']}
        save()
        for repeat in range(a.repeats):
            if repeat%2:
                candidate=run(f'r{repeat}-candidate',a.backend,protocol['tasks'],n)
                control=run(f'r{repeat}-native','native',protocol['tasks'],64)
            else:
                control=run(f'r{repeat}-native','native',protocol['tasks'],64)
                candidate=run(f'r{repeat}-candidate',a.backend,protocol['tasks'],n)
            for task in protocol['tasks']:
                report['pairs'].append(pair(control,candidate,task,repeat));save()
        for task in protocol['tasks']:
            pairs=[r for r in report['pairs'] if r['task']==task]
            report['summary'][task]=dict(pair_count=len(pairs),eligible_pairs=sum(r['budget_pair_eligible'] for r in pairs),
                all_pairs_eligible=all(r['budget_pair_eligible'] for r in pairs),
                median_cost_ratio=statistics.median(r['compute_time_ratio_candidate_over_control'] for r in pairs),
                baseline_passes=sum(r['baseline']['physical_checks_passed'] for r in pairs),
                candidate_passes=sum(r['candidate']['physical_checks_passed'] for r in pairs),
                baseline_penetration_mm=[r['baseline']['maximum_bench_penetration_mm'] for r in pairs],
                candidate_penetration_mm=[r['candidate']['maximum_bench_penetration_mm'] for r in pairs])
        report.update(status='completed',active_run=None)
    except BaseException as exc:
        report.update(status='failed',error=repr(exc));save();raise
    save()

if __name__=='__main__':main()
