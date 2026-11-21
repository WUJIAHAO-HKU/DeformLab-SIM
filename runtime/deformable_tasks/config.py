"""Versioned, strict task configuration shared by native UI and batch runner."""
import copy
import hashlib
import json
import math
from pathlib import Path

TASKS = ('cloth_grip', 'cloth_fold', 'multi_soft', 'soft_squeeze')
LABELS = ('Cloth friction grip / release', 'Cloth fold / self contact',
          'Two soft bodies / contact', 'Soft solid / compression')

def default_config(task='cloth_grip'):
    if task not in TASKS:
        raise ValueError('Unknown task: ' + str(task))
    return dict(schema_version=1, task=task, seed=0, duration_s=4.0,
                material=dict(young_pa=20000., poisson=.3, density=500., friction=.8,
                              thickness_m=.002, elasticity_damping=.05),
                geometry=dict(size_m=.08, resolution=8, cloth_resolution=20),
                solver=dict(hz=480, position_iterations=64, self_collision=True,
                            contacts=True, linear_damping=.1),
                action=dict(compression=.55, release_s=2.8, grip_gap_m=.001),
                rendering=dict(profile='studio', width=1280, height=720),
                validation=dict(penetration_tolerance_m=.004, min_volume_ratio=.02))

def validate(config):
    c=copy.deepcopy(config)
    ref=default_config(c.get('task'))
    version=c.get('schema_version')
    if version in (2,3,4,5):
        if c['task']!='cloth_fold':raise ValueError('Schema2 fixtures currently support cloth_fold only')
        ref['schema_version']=version;ref['fixtures']={'fold_support':True}
        if version==3:ref['motion']={'fold_path':'constant_radius','start_s':.3,'duration_s':2.,'terminal_clearance_m':.010}
        if version in (4,5):ref['motion']={'fold_path':'edge_normal_rotation','start_s':.3,'duration_s':2.,'terminal_edge_clearance_m':.0024}
        if version==5:ref['motion']['angular_velocity_feedforward']=True
    def keys(a,b,path):
        if not isinstance(a,dict) or set(a)!=set(b):
            raise ValueError('Missing or unknown config fields at '+path)
        for k,v in b.items():
            if isinstance(v,dict): keys(a[k],v,path+'.'+k)
    keys(c,ref,'config')
    if version not in (1,2,3,4,5): raise ValueError('Unsupported schema version')
    if version in (2,3,4,5) and type(c['fixtures']['fold_support']) is not bool:raise ValueError('fold_support must be boolean')
    def number(v,lo,hi,label,integer=False):
        if isinstance(v,bool) or not isinstance(v,(int,float)) or not math.isfinite(v) or not lo<=v<=hi:
            raise ValueError('Invalid '+label)
        if integer and (not isinstance(v,int)): raise ValueError(label+' must be integer')
    if version==3:
        motion=c['motion']
        if motion['fold_path']!='constant_radius':raise ValueError('Unknown fold trajectory')
        number(motion['start_s'],0,120,'fold start')
        number(motion['duration_s'],.1,120,'fold duration')
        number(motion['terminal_clearance_m'],.010,.07,'fold terminal clearance')
        number(c['duration_s'],.1,120,'duration')
        if motion['start_s']+motion['duration_s']>c['duration_s']:raise ValueError('Fold motion exceeds task duration')
    if version in (4,5):
        motion=c['motion']
        if motion['fold_path']!='edge_normal_rotation':raise ValueError('Unknown rotating fold trajectory')
        number(motion['start_s'],0,120,'fold start')
        number(motion['duration_s'],.1,120,'fold duration')
        number(motion['terminal_edge_clearance_m'],.0024,.01,'fold terminal edge clearance')
        number(c['duration_s'],.1,120,'duration')
        if motion['start_s']+motion['duration_s']>c['duration_s']:raise ValueError('Fold motion exceeds task duration')
    if version==5 and type(c['motion']['angular_velocity_feedforward']) is not bool:raise ValueError('Angular feedforward must be boolean')
    number(c['seed'],0,2**31-1,'seed',True)
    number(c['duration_s'],.1,120,'duration')
    m=c['material'];g=c['geometry'];s=c['solver'];a=c['action'];r=c['rendering']
    for k,lo,hi in [('young_pa',100,1e8),('poisson',0,.489),('density',1,20000),('friction',0,3),('thickness_m',.0002,.01),('elasticity_damping',0,1)]: number(m[k],lo,hi,k)
    number(g['size_m'],.02,.3,'size');number(g['resolution'],2,24,'resolution',True)
    number(g['cloth_resolution'],8,64,'cloth resolution',True)
    number(s['hz'],60,7680,'hz',True);number(s['position_iterations'],1,255,'iterations',True)
    number(s['linear_damping'],0,10,'linear damping')
    for k in ('self_collision','contacts'):
        if type(s[k]) is not bool: raise ValueError(k+' must be boolean')
    number(a['compression'],0,.8,'compression');number(a['release_s'],0,120,'release')
    number(a['grip_gap_m'],.0001,.03,'grip gap')
    if r['profile'] not in ('studio','pathtraced'): raise ValueError('Unknown rendering profile')
    number(r['width'],320,3840,'width',True);number(r['height'],240,2160,'height',True)
    number(c['validation']['penetration_tolerance_m'],0,.05,'penetration tolerance')
    number(c['validation']['min_volume_ratio'],0,1,'minimum volume ratio')
    return c

def digest(c):
    return hashlib.sha256(json.dumps(validate(c),sort_keys=True,separators=(',',':')).encode()).hexdigest()

def save_config(path,c):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(validate(c),indent=2)+'\n',encoding='utf-8')
    return path

def load_config(path):
    return validate(json.loads(Path(path).read_text(encoding='utf-8')))

def regression_configs():
    cases={t:default_config(t) for t in TASKS}
    c=default_config('cloth_grip');c['material']['friction']=0.;cases['cloth_grip_frictionless']=c
    c=default_config('cloth_fold');c['solver']['self_collision']=False;cases['cloth_fold_no_self_contact']=c
    c=default_config('multi_soft');c['solver']['contacts']=False;cases['multi_soft_no_contact']=c
    for size in (.04,.12):
        for young in (5000.,200000.):
            c=default_config('soft_squeeze');c['geometry']['size_m']=size;c['material']['young_pa']=young
            cases[f'squeeze_{size:g}m_{young:g}Pa']=c
    return cases
