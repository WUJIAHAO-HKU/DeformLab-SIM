"""Prospective held-out cases. No outcome is consulted when constructing cases."""
from pathlib import Path
import argparse,copy,hashlib,json,random

HERE=Path(__file__).resolve().parent
REMOTE='.'
CONDITIONS=[
    dict(name='narrow_soft_far',geometry_y_scale=.75,young_pa=30000.,camera_distance_m=.34),
    dict(name='wide_stiff_near',geometry_y_scale=1.25,young_pa=80000.,camera_distance_m=.27),
]
SEEDS=[202610201,202610202,202610203]
ARMS=['B','D','dense']
def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,x):Path(p).write_text(json.dumps(x,indent=2,allow_nan=False)+'\n',encoding='utf-8')
def external_dependencies(remote,builds):
    remote=Path(remote)
    dependencies=[remote/'scripts/isaac_platform_smoke.py',remote/'work/paper-generalization-v1/vision/v5/observers.py',remote/'work/paper-mechanism-v1/vision/condition_camera.py',remote/'work/paper-method-generalization-v1/reference/audit_side_push_rays.py',remote/'work/paper-generalization-v1/method/campaign_calibrated_v2.py',remote/'work/kernel-observer/physx_cp_conservative_trial.ptx']
    dependencies.extend((remote/'deformable_tasks').rglob('*.py'))
    dependencies.extend((remote/'results/paper-material-reference-v1/current-surface-coupon-v3/source/deformable_tasks').rglob('*.py'))
    for directory in ['work/native-readback','work/external-support-v3-build','work/paper-method-generalization-v1/reference/friction-sdk-v1','results/paper-material-reference-v1/current-surface-coupon-v3/source']:
        dependencies.extend(p for p in (remote/directory).iterdir() if p.is_file() and p.suffix in ['.so','.py'])
    for build in builds:
        dependencies.extend(Path(build).glob('source/*.py'))
        dependencies.extend([Path(build)/'libphysx_robot_surface.so',Path(build)/'rows.ptx'])
    dependencies.extend((HERE.parent/'budget').glob('*.py'))
    return {str(path):digest(path) for path in sorted(set(dependencies))}
def camera(template,distance):
    c=copy.deepcopy(template);c['position'][1]=-distance
    c['expected_usd_row_matrix'][3][1]=-distance
    c['calibration']['camera_to_world'][1][3]=-distance
    return c
def cases(cohort,allocation=None):
    base=json.loads((HERE/'fixture-template.json').read_text(encoding='utf-8'))
    out=[]
    if cohort not in ['development','confirmation']:raise ValueError(cohort)
    if cohort=='confirmation' and allocation is None:raise ValueError('Confirmation requires a frozen allocation receipt')
    observers=['depth'] if cohort=='development' else ['depth','truth']
    seeds=[202610200] if cohort=='development' else SEEDS
    for condition in CONDITIONS:
        for observer in observers:
            for seed in seeds:
                offset=0. if cohort=='development' else random.Random(seed).uniform(-.0025,.0025)
                order=ARMS.copy();random.Random(seed+(0 if condition['name']=='narrow_soft_far' else 100)+(0 if observer=='depth' else 10)).shuffle(order)
                for arm in order:
                    c=copy.deepcopy(base);c.update(name=f"{condition['name']}-{observer}-s{seed}-{arm}",arm=arm,observer=observer,development_smoke=cohort=='development',initial_x_offset_m=offset,geometry_y_scale=condition['geometry_y_scale'],random_initialization_seed=seed)
                    c['config']['material']['young_pa']=condition['young_pa']
                    c['config']['seed']=seed
                    selected=allocation['selected'][arm] if allocation else dict(iterations=32,component_build=f"{REMOTE}/results/paper-method-components-build-v2/"+('dense' if arm=='dense' else 'sparse'))
                    c['config']['solver']['position_iterations']=int(selected['iterations'])
                    if int(selected['iterations']) not in [16,24,32,48]:raise ValueError('Not in prospective candidate grid')
                    c['component_build']=selected['component_build']
                    c['condition']=dict(condition,camera=camera(base['condition']['camera'],condition['camera_distance_m']))
                    c['cohort']=cohort;c['budget_status']=allocation['status'] if allocation else 'development_not_budget_matched'
                    out.append(c)
    return out
def prepare(destination,cohort,allocation_path=None,promotion_path=None):
    d=Path(destination)
    if d.exists():raise FileExistsError('Never overwrite a frozen cohort '+str(d))
    allocation=json.loads(Path(allocation_path).read_text()) if allocation_path else None
    promotion=json.loads(Path(promotion_path).read_text()) if promotion_path else None
    if cohort=='confirmation':
        if not promotion or promotion.get('execution_gate_passed') is not True:raise ValueError('Six development execution gates must pass; success ranking never gates promotion')
        if not allocation or allocation.get('status') not in ['frozen_from_controlled_calibration','exploratory_shared_gpu','fixed_work_functional_only']:raise ValueError('Unknown or unmeasured budget selection')
    cc=cases(cohort,allocation);d.mkdir(parents=True)
    pins={p.name:digest(p) for p in sorted(HERE.iterdir()) if p.suffix in ['.py','.json','.txt']}
    records=[]
    for c in cc:
        p=d/(c['name']+'.json');write(p,c);records.append(dict(name=c['name'],case_file=p.name,sha256=digest(p)))
    external={}
    remote=Path(REMOTE)
    if remote.is_dir():
        external=external_dependencies(remote,{c['component_build'] for c in cc})
    protocol=dict(schema_version=1,cohort=cohort,cases=records,source_sha256=pins,external_source_sha256=external,
        method='Exact prior sparse D native library/PTX; B mass mapping and dense same-history alternative; no new law or tuned controller',
        source_root=str(HERE),conditions=CONDITIONS,random_seeds=SEEDS,seed_rule='Python random.Random(seed).uniform(-0.0025,0.0025); identical geometry offset and camera across arms',
        design='2 composite held-out conditions x 2 observers x 3 randomized initializations x 3 arms =36; initializations are paired units, not large-population coverage',
        condition_scope='Geometry aspect/contact area, Young modulus and camera distance vary together. This tests transfer to composite held-out conditions, not isolated factor attribution. No new shape topology or moving-support claim.',
        controller='Unchanged 10Hz gain1.2/s, slew5mm/s, correction4mm, same initial calibration, trajectory and target60mm; image branch never receives truth nodes',
        frozen=['controller source','observer source','target and schedule','physics dt1/960s','material model/Poisson .3/density1000/friction .4/damping0','collision offsets','mesh topology','render profile/resolution','evaluation windows','method binaries'],
        primary='Full-physics-sample mean absolute material-front boundary error, t2.1--2.4s, same x60mm target for image/truth; all frames required',
        secondary=['free t2.8--3.2 target error and drift','actual tool clearance>1mm throughout free interval','support penetration/domain','min tetJ','invalid observations and ray identity','actual method invocation/row activity','timing scope + GPU contamination'],
        success='Original2mm MAE threshold retained as auxiliary; no threshold-shopping to manufacture failures or gains',
        analysis='Report each paired difference, all36 outcomes, descriptive paired condition/observer summaries; only3 independent starts per cell, no population generalization confidence claim',
        outcome_policy='Run every planned case once. Keep task/material/domain failures. Infrastructure/source/count/clock/replay failure halts and is retained. No case replacement or outcome-dependent retuning.',
        isolation='Six separate development cases, seed202610200 offset0. Their task rankings never select protocol. Only execution-validity gates promote.',
        budget=allocation or dict(status='development_not_budget_matched'),allocation_sha256=digest(allocation_path) if allocation_path else None,
        promotion_sha256=digest(promotion_path) if promotion_path else None,
        timing='No per-round component bin; retain exact counter evidence. Physics instrumentation and full-loop times separately; only uncontaminated certified calibration may support same-wall-budget claim.',
        resources=dict(native_timeout_s=360,min_gpu_free_mib=12288,max_case_scratch_mib=512,scratch='root-selected per-case /dev/shm; copy + SHA verify on E before any raw cleanup',estimated_gpu_runtime_s_per_case='Existing3.2s cases under300s watchdog; new timing must be measured',expected_payload='33 RGB-D frames plus3073x125 nodal positions/velocities and counter arrays; verify actual bytes during smoke'))
    write(d/'protocol.json',protocol)
    return dict(protocol=str(d/'protocol.json'),sha256=digest(d/'protocol.json'),cases=len(cc),status='prepared_not_executed')
def verify(folder):
    d=Path(folder);p=json.loads((d/'protocol.json').read_text())
    for n,h in p['source_sha256'].items():assert digest(HERE/n)==h,('changed_source',n)
    for n,h in p.get('external_source_sha256',{}).items():assert digest(n)==h,('changed_external_source',n)
    for c in p['cases']:assert digest(d/c['case_file'])==c['sha256']
    return p
if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('action',choices=['prepare','verify']);ap.add_argument('--out',required=True);ap.add_argument('--cohort',choices=['development','confirmation'],default='development');ap.add_argument('--allocation');ap.add_argument('--promotion');a=ap.parse_args()
    if a.action=='prepare':print(json.dumps(prepare(a.out,a.cohort,a.allocation,a.promotion)))
    else:print(json.dumps(dict(pins_valid=True,cases=len(verify(a.out)['cases']))))
