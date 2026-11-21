"""Explicit, restricted multi-volume CP mapping diagnostic; no material acceptance."""
import csv,json,math,os
from pathlib import Path
from .cone_backend import sha
MODES={'observe':0,'repair':1}
MAPPED_BACKENDS=('ss-cp-map-native','ss-cp-map-common','volume-mapping')
OBSERVE_BACKEND='ss-cp-observe-native'
CP_DIAGNOSTIC_BACKENDS=MAPPED_BACKENDS+(OBSERVE_BACKEND,)
def requested(task,backend):
    value=os.environ.get('DEFORMABLE_CP_MAPPING','0')
    if backend=='volume-mapping' and task=='robot_volume':
        if value!='0' or os.environ.get('DEFORMABLE_CONE_TRACE','0')!='0':raise ValueError('Robot mapping requires its named bridge without two-body observer switches')
        return None
    if backend==OBSERVE_BACKEND:
        if task!='multi_soft' or value not in ('0','observe') or os.environ.get('DEFORMABLE_CONE_TRACE','0') not in ('0','1'):
            raise ValueError('Named CP observer requires multi_soft and mapping disabled; no fallback')
        return 0
    if backend in MAPPED_BACKENDS:
        if task!='multi_soft' or value not in ('0','repair') or os.environ.get('DEFORMABLE_CONE_TRACE','0') not in ('0','1'):
            raise ValueError('Named mapped backends require two-volume multi_soft and the repair mode; no fallback')
        return 1
    if value!='0' and (value not in MODES or task!='multi_soft' or backend not in ('ss-cone-plane','ss-cone-control') or os.environ.get('DEFORMABLE_CONE_TRACE','0')!='1'):
        raise ValueError('CP mapping requires explicit observe/repair, multi_soft, an explicit cone/control binding and cone trace; no fallback')
    return MODES.get(value)
def extend_artifacts(root,files):
    root=Path(root);build=root/'results/multisoft-cp-mapping-build-v2'
    files.update(library=build/'libmultisoft_cp_mapping.so',manifest=build/'build.json',cone_ptx=build/'cp-mapping.ptx',
        mapping_selftest=root/'results/multisoft-cp-mapping-selftest-v2/summary.json',
        mapping_compat=root/'results/multisoft-cp-mapping-compat-v2/summary.json',
        mapping_dispatch=root/'results/multisoft-cp-mapping-dispatch-v2/summary.json',
        mapping_trace=root/'results/multisoft-cp-mapping-trace-v2/summary.json',
        mapping_native_average=root/'results/multisoft-cp-mapping-native-average-v1/summary.json')
    for key in ('manifest','mapping_selftest','mapping_compat','mapping_dispatch','mapping_trace','mapping_native_average'):
        record=json.loads(files[key].read_text())
        if key=='manifest':
            if record['status']!='compiled' or sha(files['library'])!=record['library_sha256']:raise ValueError('Changed mapping build')
        elif not record['passed']:raise ValueError('Mapping prerequisite failed')
        for name,digest in record['input_sha256'].items():
            if sha(name)!=digest:raise ValueError('Changed mapping input '+name)
    return files
def evidence(path,mode,values,status,native_averages):
    path=Path(path);lines=path.read_text().splitlines();rows=list(csv.DictReader([l for l in lines if not l.startswith('#')],delimiter='\t'))
    cycles,partitions,published,errors,bodies,scratch=map(int,values);groups={}
    for row in rows:groups.setdefault(int(row['cycle']),[]).append(row)
    coverage=sorted(groups)==list(range(1,published+1)) and all(len(rs)==2 and len({r['body'] for r in rs})==2 for rs in groups.values())
    finite=all(math.isfinite(float(v)) for r in rows for v in r.values())
    mode_valid=all(int(r['mode'])==mode and int(r['min_copies'])>=1 and int(r['max_copies'])>=int(r['min_copies']) and float(r['invdt'])>0 for r in rows)
    footer=bool(lines and lines[-1]==f'# cycles={cycles} partitions={partitions} publications={published} errors={errors}')
    partition_coverage=sum(int(rs[0]['partitions']) for rs in groups.values())==partitions
    verified=status==errors==0 and cycles==published==native_averages and cycles>0 and bodies==2 and coverage and finite and mode_valid and footer and partition_coverage
    report=dict(mode=mode,cycles=cycles,partition_calls=partitions,publication_calls=published,errors=errors,bodies=bodies,scratch_bytes=scratch,exit_code=status,
        ledger=str(path),ledger_sha256=sha(path),ledger_rows=len(rows),execution_verified=bool(verified),mapping_only=True,
        material_law='native_unchanged',intrusive_synchronization=True,physical_improvement_accepted=False,performance_accepted=False,platform_acceptance=False)
    for phase in ('native','after'):
        residuals=[math.sqrt(sum((float(r[phase+'_p'+k])-float(r['before_p'+k]))**2 for k in 'xyz')) for r in rows]
        report[phase+'_maximum_internal_momentum_change_kg_m_s']=max(residuals,default=0.)
        report[phase+'_sum_internal_momentum_change_by_body_kg_m_s']={b:[sum(float(r[phase+'_p'+k])-float(r['before_p'+k]) for r in rows if r['body']==b) for k in 'xyz'] for b in sorted({r['body'] for r in rows})}
    return report

def bridge_artifacts(root):
    root=Path(root);receipt=root/'results/cp-mapping-bridge-equivalence-v1/summary.json'
    record=json.loads(receipt.read_text())
    if not record.get('passed') or not record.get('helper_outputs_byte_identical'):raise ValueError('Verified shared CP mapping arithmetic required')
    for name,digest in record['input_sha256'].items():
        if sha(name)!=digest:raise ValueError('Changed bridge equivalence input '+name)
    return dict(equivalence=receipt,library=root/'work/kernel-observer/libphysx_cp_trial_hook.so',ptx=root/'work/kernel-observer/physx_cp_conservative_trial.ptx')
