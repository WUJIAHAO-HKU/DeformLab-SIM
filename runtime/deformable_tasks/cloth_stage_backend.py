"""Explicit bounded cloth diagnostics; no candidate or performance acceptance."""
import ctypes,hashlib,json,os,shutil
from pathlib import Path
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        while chunk:=f.read(4*1024*1024):h.update(chunk)
    return h.hexdigest()
def requested(task,backend):
    mode=os.environ.get('DEFORMABLE_CLOTH_STAGES','0')
    contacts=os.environ.get('DEFORMABLE_CLOTH_CONTACTS','0')
    bench=os.environ.get('DEFORMABLE_CLOTH_BENCH_WINDOW','0')
    if bench not in ('0','1') or (bench=='1' and (mode!='1' or contacts!='0' or task!='cloth_grip' or backend!='native')):raise ValueError('Bench window requires explicit native cloth_grip stages without contact-row capture')
    if contacts not in ('0','1') or (contacts=='1' and mode!='1'):raise ValueError('Cloth contact records require explicit cloth stages')
    if mode not in ('0','1'):raise ValueError('Cloth stage observer must be explicitly 0 or 1')
    if mode=='0':return False
    if task not in ('cloth_grip','cloth_fold') or backend!='native':raise ValueError('Cloth stages support only native single-cloth tasks; no fallback')
    if os.environ.get('DEFORMABLE_KERNEL_PROFILE','0')!='1':raise ValueError('Cloth stages require explicit kernel profiling')
    if any(os.environ.get(k,'0')!='0' for k in ('DEFORMABLE_MULTISOFT_STATES','DEFORMABLE_MULTISOFT_HISTORY','DEFORMABLE_CONE_TRACE','DEFORMABLE_CP_MAPPING')):raise ValueError('Cloth stages cannot compose with volume observers or repairs')
    return True
def artifacts(root):
    root=Path(root);contacts=os.environ.get('DEFORMABLE_CLOTH_CONTACTS','0')=='1';bench=os.environ.get('DEFORMABLE_CLOTH_BENCH_WINDOW','0')=='1'
    build=root/('results/cloth-bench-observer-build-v1' if bench else 'results/cloth-contact-observer-build-v3' if contacts else 'results/cloth-stage-observer-build-v1')
    files=dict(library=build/('libcloth_contact_observer.so' if contacts else 'libcloth_stage_observer.so'),manifest=build/'build.json',
        cloth_stage_selftest=root/('results/cloth-bench-observer-selftest-v1/summary.json' if bench else 'results/cloth-contact-observer-selftest-v3/summary.json' if contacts else 'results/cloth-stage-observer-selftest-v1/summary.json'),
        selftest=root/('results/cloth-bench-dispatch-selftest-v1/summary.json' if bench else 'results/cloth-contact-dispatch-selftest-v3/summary.json' if contacts else 'results/cloth-stage-dispatch-selftest-v1/summary.json'))
    manifest=json.loads(files['manifest'].read_text())
    if manifest.get('status')!='compiled' or sha(files['library'])!=manifest['library_sha256']:raise ValueError('Verified cloth stage build required')
    pins=dict(manifest['input_sha256'])
    for key in ('selftest','cloth_stage_selftest'):
        test=json.loads(files[key].read_text())
        if not test.get('passed'):raise ValueError('Passing cloth owned/dispatch tests required')
        pins.update(test['input_sha256'])
    if any(sha(p)!=h for p,h in pins.items()):raise ValueError('Cloth stage artifact changed')
    return files
class ClothStages:
    def __init__(self,lib,root,dest,extended_window=False):
        self.lib=lib;self.dest=Path(dest);self.stopped=False
        self.contacts=os.environ.get('DEFORMABLE_CLOTH_CONTACTS','0')=='1'
        self.rigid=os.environ.get('DEFORMABLE_CLOTH_RIGID_ROWS','0')=='1'
        self.generation=os.environ.get('DEFORMABLE_CLOTH_GENERATION','0')=='1'
        self.self_rows=os.environ.get('DEFORMABLE_CLOTH_SELF_ROWS','0')=='1'
        if self.self_rows and (not extended_window or self.rigid or self.contacts or self.generation or os.environ.get('DEFORMABLE_CLOTH_SPLIT_STAGES','0')!='1'):raise ValueError('Self rows require exclusive verified split observer')
        if self.generation and not self.rigid:raise ValueError('Generation capture requires rigid rows')
        if self.rigid and (not extended_window or self.contacts or os.environ.get('DEFORMABLE_CLOTH_SPLIT_STAGES','0')!='1'):raise ValueError('Rigid rows require the verified split observer')
        self.suffixes=('.bin','.jsonl','.self.bin') if self.self_rows else ('.bin','.jsonl','.rigid.bin') if self.rigid else ('.bin','.jsonl','.extra.bin') if self.contacts else ('.bin','.jsonl')
        self.first=int(os.environ.get('DEFORMABLE_CLOTH_FIRST','1'));self.last=int(os.environ.get('DEFORMABLE_CLOTH_LAST','2'))
        self.bench=bool(extended_window) or os.environ.get('DEFORMABLE_CLOTH_BENCH_WINDOW','0')=='1'
        if extended_window not in (False,32,64):raise ValueError('Unsupported explicit extended window')
        limit=int(extended_window) if extended_window else 32 if self.bench else 4
        if not 1<=self.first<=self.last or self.last-self.first>=limit:raise ValueError(f'Cloth window must contain 1 to {limit} complete steps')
        archive=Path('./evidence-archive/cloth-stage-captures')
        archive.mkdir(parents=True,exist_ok=True)
        required_gib=17 if self.self_rows else 11 if self.rigid else 9 if limit==64 else 5 if self.bench else 3
        if shutil.disk_usage(archive).free<required_gib*1024**3:raise RuntimeError(f'Cloth observation archive needs {required_gib} GiB free')
        relative=self.dest.resolve().relative_to((Path(root)/'results').resolve())
        target=archive/relative;target.mkdir(parents=True,exist_ok=False);self.prefix=target/'cloth-stages'
        self.lib.cloth_stage_start.argtypes=[ctypes.c_char_p,ctypes.c_uint,ctypes.c_uint]
        self.lib.cloth_stage_stop.argtypes=[ctypes.POINTER(ctypes.c_ulonglong)]
        for suffix in self.suffixes:
            if os.path.lexists(self.dest/('cloth-stages'+suffix)):raise ValueError('Cloth evidence already exists')
        if self.lib.cloth_stage_start(str(self.prefix).encode(),self.first,self.last):raise RuntimeError('Cannot start cloth stages')
        try:
            for suffix in self.suffixes:(self.dest/('cloth-stages'+suffix)).symlink_to(Path(str(self.prefix)+suffix))
        except BaseException:
            values=(ctypes.c_ulonglong*6)();self.lib.cloth_stage_stop(values);raise
    def finish(self,steps):
        if self.stopped:raise RuntimeError('Cloth observer already stopped')
        values=(ctypes.c_ulonglong*6)();status=self.lib.cloth_stage_stop(values);self.stopped=True
        files=[self.dest/('cloth-stages'+s) for s in self.suffixes]
        evidence=dict(bench_window=self.bench,window=[self.first,self.last],steps=int(values[0]),state_records=int(values[1]),kernel_pairs=int(values[2]),raw_bytes=int(values[3]),nodes=int(values[4]),errors=int(values[5]),exit_code=status,
            read_only=True,intrusive_synchronization=True,repair_execution_verified=False,performance_acceptance=False,platform_acceptance=False)
        evidence['files_sha256']={str(p):sha(p) for p in files}
        evidence['storage_paths']={str(p):str(p.resolve()) for p in files}
        evidence['capture_valid']=status==0 and values[5]==0 and values[0]==steps and steps>=self.last and values[1]>0 and values[1]==2*values[2] and values[4]>0 and files[0].stat().st_size==values[3]
        if self.contacts:
            self.lib.cloth_contact_extra_bytes.restype=ctypes.c_ulonglong
            extra=int(self.lib.cloth_contact_extra_bytes());evidence['extra_bytes']=extra
            evidence['contact_parameters_captured']=True
            evidence['capture_valid'] &= extra>0 and files[2].stat().st_size==extra
        if self.rigid:
            self.lib.cloth_rigid_extra_bytes.restype=ctypes.c_ulonglong
            extra=int(self.lib.cloth_rigid_extra_bytes());evidence['rigid_extra_bytes']=extra
            evidence['rigid_contact_parameters_captured']=True
            evidence['capture_valid'] &= extra>0 and files[2].stat().st_size==extra
        if self.self_rows:
            self.lib.cloth_self_extra_bytes.restype=ctypes.c_ulonglong
            extra=int(self.lib.cloth_self_extra_bytes());evidence['self_extra_bytes']=extra
            counts={}
            with files[1].open() as source:
                for line in source:
                    if '"self_sections"' not in line:continue
                    record=json.loads(line);key=(record['kernel'],record['phase']);counts[key]=counts.get(key,0)+1
            names=('cloth_queryClothClothContactVTCountLaunch','cloth_queryClothClothContactEECountLaunch','cloth_solveClothClothDeltaVTLaunch','cloth_solveClothClothDeltaEELaunch')
            evidence['self_contact_parameters_captured']=True
            evidence['self_state_records']={name:{phase:counts.get((name,phase),0) for phase in ('before','after')} for name in names}
            evidence['capture_valid'] &= extra>0 and files[2].stat().st_size==extra and all(counts.get((name,'before'),0)>0 and counts.get((name,'before'))==counts.get((name,'after')) for name in names)
        if self.generation:
            with files[1].open() as source:
                records=[json.loads(line) for line in source if '"generation_sections"' in line]
            before=sum(r['phase']=='before' for r in records);after=sum(r['phase']=='after' for r in records)
            evidence.update(generation_parameters_captured=True,generation_state_records=len(records))
            evidence['capture_valid'] &= before>0 and before==after
        return evidence
