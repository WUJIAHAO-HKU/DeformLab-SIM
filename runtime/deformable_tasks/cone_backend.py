"""Restricted auxiliary-plane cone experiments, never general contact acceptance."""
import csv,hashlib,json,math,os
from pathlib import Path
CONE_BACKENDS={'ss-cone-control':0,'ss-cone-plane':1,'ss-cp-observe-native':0,'ss-cp-map-native':0,'ss-cp-map-common':0,'volume-mapping':0}
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        while data:=f.read(4*1024*1024):h.update(data)
    return h.hexdigest()
def artifacts(root,mapping=False):
    from .ss_common_backend import artifacts as common_artifacts
    root=Path(root);files=common_artifacts(root);build=root/'results/multisoft-cone-binding-build-v2'
    files.update(library=build/'libmultisoft_cone_binding.so',manifest=build/'build.json',
        cone_ptx=build/'cone-ledger.ptx',split_ptx=root/'results/multisoft-velocity-cone-native-forcing-v1/native-and-split.ptx',
        cone_selftest=root/'results/multisoft-cone-binding-selftest-v1/summary.json',
        cone_common_selftest=root/'results/multisoft-cone-common-compat-v1/summary.json',
        cone_dispatch_selftest=root/'results/multisoft-cone-dispatch-v1/summary.json')
    for key in ('manifest','cone_selftest','cone_common_selftest','cone_dispatch_selftest'):
        record=json.loads(files[key].read_text())
        if key=='manifest':
            if record['status']!='compiled' or sha(files['library'])!=record['library_sha256']:raise ValueError('Cone binding changed')
        elif not record['passed']:raise ValueError('Cone prerequisite failed')
        for name,digest in record['input_sha256'].items():
            if sha(name)!=digest:raise ValueError('Changed cone input '+name)
    if mapping or os.environ.get('DEFORMABLE_CONE_TRACE','0')=='1':
        trace=root/'results/multisoft-cone-trace-build-v2'
        files.update(library=trace/'libmultisoft_cone_trace.so',manifest=trace/'build.json',cone_ptx=trace/'cone-trace.ptx',
            trace_selftest=root/'results/multisoft-cone-trace-selftest-v2/summary.json',
            trace_compat_selftest=root/'results/multisoft-cone-trace-compat-v1/summary.json',
            trace_dispatch_selftest=root/'results/multisoft-cone-trace-dispatch-v1/summary.json')
        for key in ('manifest','trace_selftest','trace_compat_selftest','trace_dispatch_selftest'):
            record=json.loads(files[key].read_text())
            if key=='manifest':
                if record['status']!='compiled' or sha(files['library'])!=record['library_sha256']:raise ValueError('Trace binding changed')
            elif not record['passed']:raise ValueError('Trace prerequisite failed')
            for name,digest in record['input_sha256'].items():
                if sha(name)!=digest:raise ValueError('Changed trace input '+name)
    if mapping or os.environ.get('DEFORMABLE_CP_MAPPING','0')!='0':
        from .cp_mapping_backend import extend_artifacts
        files=extend_artifacts(root,files)
    return files

def evidence(path,mode,values,status,native_calls):
    path=Path(path);lines=path.read_text().splitlines();rows=list(csv.DictReader([l for l in lines if not l.startswith('#')],delimiter='\t'))
    calls,published,rejected,forced,errors,bodies,scratch=map(int,values)
    summary=dict(mode=mode,calls=calls,publication_calls=published,rejected_calls=rejected,
        forced_velocity_calls=forced,errors=errors,bodies=bodies,scratch_bytes=scratch,exit_code=status,
        ledger=str(path),ledger_sha256=sha(path),ledger_rows=len(rows),
        volume_projections=sum(int(r['volume_projections']) for r in rows),
        plane_projections=sum(int(r['plane_projections']) for r in rows),
        auxiliary_plane_impulse_kg_m_s=[sum(float(r['floor_p'+k]) for r in rows) for k in 'xyz'],
        guard_kinetic_change_J=sum(float(r['guard_dKE']) for r in rows),
        native_velocity_stage_kinetic_change_J=sum(float(r['native_velocity_dKE']) for r in rows),
        maximum_internal_momentum_residual=max((math.sqrt(sum(float(r['internal_p'+k])**2 for k in 'xyz')) for r in rows),default=0.),
        maximum_internal_angular_residual=max((math.sqrt(sum(float(r['internal_l'+k])**2 for k in 'xyz')) for r in rows),default=0.),
        maximum_work_ledger_residual_J=max((abs(float(r['guard_dKE'])-float(r['internal_work'])-float(r['floor_work'])) for r in rows),default=0.))
    grouped={}
    for r in rows:grouped.setdefault(int(r['call']),[]).append(r)
    coverage=sorted(grouped)==list(range(1,published+1)) and all(len(rs)==2 and len({r['body'] for r in rs})==2 for rs in grouped.values())
    footer=bool(lines and lines[-1]==f'# calls={calls} publications={published} rejections={rejected} errors={errors}')
    modes=all(int(r['mode'])==mode and int(r['forcing']) in (0,1) and int(r['step'])>0 and float(r['h'])>0 for r in rows)
    finite=all(math.isfinite(float(v)) for r in rows for v in r.values())
    valid=all(float(r['guard_dKE'])<=1e-12 and float(r['published_guard_dKE'])<=1e-12 and float(r['impulse_l1_residual'])<=1e-10 and float(r['floor_pz'])>=0 and float(r['floor_px'])==0 and float(r['floor_py'])==0 for r in rows)
    active=mode==0 or summary['plane_projections']+summary['volume_projections']>0
    summary['execution_verified']=bool(status==0 and errors==0 and rejected==0 and calls==published==native_calls and calls>0 and bodies==2 and len(rows)==2*calls and coverage and footer and modes and finite and valid and active and summary['maximum_internal_momentum_residual']<=1e-10 and summary['maximum_internal_angular_residual']<=1e-11 and summary['maximum_work_ledger_residual_J']<=1e-12)
    summary.update(intrusive_synchronization=True,auxiliary_plane_only=True,
        native_contact_manifold_reaction_accepted=False,physical_improvement_accepted=False,
        full_force_work_accepted=False,performance_accepted=False,platform_acceptance=False)
    return summary
