"""SDK readback checks, separate from the frozen D solver and controller."""
import numpy as np

def effective_friction(rigid_dynamic, soft_dynamic, mode):
    r=float(rigid_dynamic);s=float(soft_dynamic)
    if not np.isfinite([r,s]).all() or min(r,s)<0:raise ValueError('Invalid native coefficient')
    if mode==0:return (r+s)/2
    if mode==1:return min(r,s)
    if mode==2:return r*s
    if mode==3:return max(r,s)
    raise ValueError('Unknown native combine code')

def validate(raw,tool_center):
    soft=raw['soft_materials'];assert len(soft)==1
    smu=float(soft[0]['dynamic_friction']);assert abs(smu-.4)<1e-7
    assert raw['soft_shape_pointer'] and soft[0]['material_pointer']
    selected={}
    for name,center,half,art in [('bench',[0,0,.085],[.08,.08,.035],False),('tool',tool_center,[.04,.04,.01],True)]:
        matches=[s for s in raw['rigid_shapes'] if s['simulation_shape'] and s['articulation_link']==art
                 and np.allclose(s.get('half_extents_m',[0,0,0]),half,rtol=0,atol=2e-7)
                 and np.allclose(s['shape_world_pose']['position'],center,rtol=0,atol=2e-6)]
        assert len(matches)==1,(name,'shape identity',len(matches))
        shape=matches[0];assert shape['shape_pointer'] and shape['actor_pointer']
        assert len(shape['materials'])==1
        m=shape['materials'][0];assert m['material_pointer']
        assert abs(float(m['dynamic_friction'])-.4)<1e-7 and abs(float(m['static_friction'])-.4)<1e-7
        mu=effective_friction(m['dynamic_friction'],smu,m['friction_combine_mode'])
        assert .1599999<=mu<=.4000001
        selected[name]=dict(shape=shape,effective_dynamic_coefficient=mu,
                            combine_rule={0:'average',1:'minimum',2:'multiply',3:'maximum'}[m['friction_combine_mode']])
    return dict(record_valid=True,raw_sdk=raw,selected=selected,
        scope='Actual SDK shape/material bindings and coefficient combination decoded from matching native solve. No GPU row-material index or friction-force correctness claim.',
        soft_static_coefficient_observed=False,gpu_row_material_indices_observed=False)

def test():
    import copy
    raw=dict(soft_shape_pointer=1,soft_materials=[dict(dynamic_friction=.4,material_pointer=2)],rigid_shapes=[])
    for name,center,half,art in [('bench',[0,0,.085],[.08,.08,.035],False),('tool',[-.061,0,.14],[.04,.04,.01],True)]:
        raw['rigid_shapes'].append(dict(simulation_shape=True,articulation_link=art,half_extents_m=half,
            shape_world_pose=dict(position=center),shape_pointer=3,actor_pointer=4,
            materials=[dict(dynamic_friction=.4,static_friction=.4,friction_combine_mode=0,material_pointer=5)]))
    count=0
    for mode in range(4):
        q=copy.deepcopy(raw)
        for shape in q['rigid_shapes']:shape['materials'][0]['friction_combine_mode']=mode
        r=validate(q,[-.061,0,.14]);assert r['record_valid'];count+=1
    for change in range(7):
        q=copy.deepcopy(raw)
        if change==0:q['soft_materials'][0]['dynamic_friction']=0
        if change==1:q['rigid_shapes'][0]['materials'][0]['friction_combine_mode']=8
        if change==2:q['rigid_shapes'][1]['shape_world_pose']['position'][0]=0
        if change==3:q['rigid_shapes'][0]['materials'][0]['dynamic_friction']=0
        if change==4:q['rigid_shapes'][1]['materials'][0]['static_friction']=0
        if change==5:q['rigid_shapes'].append(copy.deepcopy(q['rigid_shapes'][0]))
        if change==6:q['soft_materials'][0]['material_pointer']=0
        try:validate(q,[-.061,0,.14])
        except (AssertionError,ValueError):count+=1
        else:raise AssertionError('Mutation not rejected '+str(change))
    return dict(passed=True,tests=count,scope='Python schema/combine checks; SDK getter requires live smoke')
