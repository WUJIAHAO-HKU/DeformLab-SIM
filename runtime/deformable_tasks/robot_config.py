"""Strict robot task configuration and explicit adapter dispatch for the shared entry."""
import copy, hashlib, json, math
from pathlib import Path
from .weight_backend import WEIGHT_BACKENDS
from .repair_contract import selection
from .robot_surface_backend import BACKENDS as ROBOT_SURFACE_BACKENDS
ROBOT_BACKENDS=('native','cp-control','cp-sequential','volume-mapping')+WEIGHT_BACKENDS+ROBOT_SURFACE_BACKENDS
def default_robot_config():
    return dict(schema_version=1,task='robot_volume',duration_s=7.5,
        material=dict(young_pa=50000.,poisson=.3,density=1000.,friction=.5,elasticity_damping=.05),
        geometry=dict(size_m=.04,resolution=4,yaw_deg=0.,offset_x_m=0.),
        solver=dict(hz=960,position_iterations=32,contact_offset_m=.001,speculative_ccd=False,zero_settling=True,solver_type='TGS',external_forces_every_iteration=True),
        action=dict(grip_mode='close',robot_collision=True,lift_height_m=.1,lift_profile='quintic',hold_arm_target='feedback'),
        measurement=dict(finger_wrenches=True),
        validation=dict(maximum_tabletop_penetration_m=.001,minimum_volume_ratio=.001))
def validate_robot_config(config):
    c=copy.deepcopy(config);ref=default_robot_config()
    def shape(a,b,path):
        if not isinstance(a,dict) or set(a)!=set(b):raise ValueError('Missing/unknown robot fields: '+path)
        for k,v in b.items():
            if isinstance(v,dict):shape(a[k],v,path+'.'+k)
    shape(c,ref,'config')
    if type(c['schema_version']) is not int or c['schema_version']!=1 or c['task']!='robot_volume':raise ValueError('Invalid robot schema/task')
    def number(v,lo,hi,label,integer=False):
        if type(v) not in (int,float) or not math.isfinite(v) or not lo<=v<=hi or (integer and type(v) is not int):raise ValueError('Invalid '+label)
    number(c['duration_s'],7.5,7.5,'robot duration')
    m=c['material'];g=c['geometry'];s=c['solver'];a=c['action']
    for key,lo,hi in [('young_pa',100.,1e8),('poisson',0.,.489),('density',1.,20000.),('friction',0.,3.)]:number(m[key],lo,hi,key)
    if type(m['elasticity_damping']) not in (int,float) or m['elasticity_damping'] not in (0.,.05):raise ValueError('Unsupported robot damping')
    number(g['size_m'],.02,.06,'gripper-compatible size');number(g['resolution'],2,8,'resolution',True)
    number(g['yaw_deg'],-30.,30.,'yaw');number(g['offset_x_m'],-.008,.008,'offset')
    number(s['hz'],240,960,'hz',True)
    if s['hz'] not in (240,480,960):raise ValueError('Unsupported robot hz')
    number(s['position_iterations'],1,255,'iterations',True);number(s['contact_offset_m'],.00001,.02,'contact offset')
    for v in [s['speculative_ccd'],s['zero_settling'],s['external_forces_every_iteration'],a['robot_collision'],c['measurement']['finger_wrenches']]:
        if type(v) is not bool:raise ValueError('Expected boolean robot option')
    if not s['zero_settling']:raise ValueError('Unified comparison requires zero settling on every backend')
    if s['solver_type']!='TGS' or not s['external_forces_every_iteration']:raise ValueError('Unified comparison requires matched TGS and external forces every iteration')
    if a['grip_mode'] not in ('close','open') or a['lift_profile']!='quintic' or a['hold_arm_target'] not in ('feedback','fixed'):raise ValueError('Unsupported robot action')
    number(a['lift_height_m'],.09,.2,'lift height')
    number(c['validation']['maximum_tabletop_penetration_m'],0.,.001,'table penetration')
    number(c['validation']['minimum_volume_ratio'],.001,1.,'volume ratio')
    return c
def digest_robot(c):
    return hashlib.sha256(json.dumps(validate_robot_config(c),sort_keys=True,separators=(',',':')).encode()).hexdigest()
def load_robot_config(path):
    return validate_robot_config(json.loads(Path(path).read_text()))
def save_robot_config(path,c):
    p=Path(path);p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(validate_robot_config(c),indent=2)+'\n');return p
def robot_plan(config,backend):
    c=validate_robot_config(config)
    if backend in ROBOT_SURFACE_BACKENDS:
        from .robot_surface_backend import ROUTES,plan as surface_plan
        base=ROUTES[backend][0]
        return c,surface_plan(c,backend,robot_plan(c,base)[1],Path(__file__).resolve().parents[1])
    if backend not in ROBOT_BACKENDS:raise ValueError('Robot adapter does not support backend '+backend+'; no native fallback')
    if backend!='native' and (c['solver']['hz']!=960 or c['solver']['position_iterations'] not in (16,32,64) or c['geometry']['resolution']!=4):
        raise ValueError('Robot CP bridge requires 960 Hz, 16/32/64 iterations and resolution 4 until partition generalization is verified')
    trial='pass-through' if backend in WEIGHT_BACKENDS else {'native':'none','cp-control':'pass-through','cp-sequential':'split-sequential','volume-mapping':'split-velocity'}[backend]
    mapping_files={}
    if backend=='volume-mapping':
        from .cp_mapping_backend import bridge_artifacts
        mapping_files=bridge_artifacts(Path(__file__).resolve().parents[1])
    return c,dict(name=backend,task='robot_volume',mapping_artifacts={k:str(v) for k,v in mapping_files.items()},trial_mode=trial,config_sha256=digest_robot(c),
        selected_components=(selection(backend)['selected_components'] if backend in WEIGHT_BACKENDS or backend=='volume-mapping' else ['single_volume_mass_velocity','sequential_material_damping'] if backend=='cp-sequential' else []),
        execution_verified=False,latest_contact_candidate_selected=False,
        scope=('Articulated single-volume fused contact weight experiment; material solver unchanged, candidate-v5 not enabled.' if backend in WEIGHT_BACKENDS else 'Actual articulated Franka. Existing material repair only; candidate-v5 contact repair is not routed here.'),
        numerical_diagnostics=True,performance_acceptance=False)
def recorder_arguments(config_path,config,backend,render=False):
    if backend in ROBOT_SURFACE_BACKENDS:
        from .robot_surface_backend import ROUTES
        robot_plan(config,backend)
        base,mode=ROUTES[backend]
        return recorder_arguments(config_path,config,base,render)+['--surface-runtime',mode]
    c,meta=robot_plan(config,backend);a=c['action']
    return ['--unified-task-config',str(config_path),'--unified-backend',backend,
            '--robot-collision','on' if a['robot_collision'] else 'off',
            '--cp-conservative-trial',meta['trial_mode'],
            '--zero-settling'] + ([] if render else ['--no-captures'])

