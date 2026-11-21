"""Explicit read-only launch census; no repair, active-row, or timing acceptance."""
import csv,ctypes,hashlib,json,os,re,shutil
from pathlib import Path
from .ss_common_backend import COMMON_BACKENDS,artifacts as common_artifacts
from .cone_backend import CONE_BACKENDS,artifacts as cone_artifacts,evidence as cone_evidence_report
from .cp_mapping_backend import CP_DIAGNOSTIC_BACKENDS
TASKS=('cloth_grip','cloth_fold','multi_soft')
def sha(p):
    digest=hashlib.sha256()
    with Path(p).open('rb') as source:
        while chunk:=source.read(4*1024*1024):digest.update(chunk)
    return digest.hexdigest()
STATE_ARCHIVE=Path('./evidence-archive/multisoft-state-captures')
STATE_SUFFIXES=('.jsonl','.bin.gz')
def snapshot_prefix(root,dest,archive_root=STATE_ARCHIVE):
    """Reserve a fresh archive directory; primary results keep transparent links."""
    results=(Path(root)/'results').resolve();dest=Path(dest).resolve()
    relative=dest.relative_to(results)
    if not relative.parts:raise ValueError('Snapshots require a case directory within results')
    if not dest.is_dir():raise ValueError('Snapshot case directory does not exist')
    for suffix in STATE_SUFFIXES:
        if os.path.lexists(dest/('native-states'+suffix)):raise ValueError('Snapshot evidence already exists')
    archive_root=Path(archive_root);archive_root.mkdir(parents=True,exist_ok=True)
    archive_root=archive_root.resolve();target=archive_root/relative
    target.resolve().relative_to(archive_root)
    # The native observer permits 4 GiB of raw data plus index/compression overhead.
    if shutil.disk_usage(archive_root).free<5*1024**3:raise RuntimeError('Snapshot archive needs 5 GiB of free space')
    target.mkdir(parents=True,exist_ok=False)
    return target/'native-states'
def requested(task,backend):
    from .cloth_split_backend import requested as split_requested
    if split_requested(task,backend):return True
    from .cloth_stage_backend import requested as cloth_requested
    cloth_requested(task,backend)
    from .cp_mapping_backend import requested as mapping_requested
    mapping_requested(task,backend)
    from .batched_cp_backend import requested as batch_requested
    if batch_requested(task,backend):return True
    if backend=='volume-mapping' and task=='robot_volume':
        if any(os.environ.get(k,'0')!='0' for k in ('DEFORMABLE_KERNEL_PROFILE','DEFORMABLE_MULTISOFT_STATES','DEFORMABLE_MULTISOFT_HISTORY')):
            raise ValueError('Robot volume mapping uses its explicit single-body bridge; no mixed profiler fallback')
        return False
    trace=os.environ.get('DEFORMABLE_CONE_TRACE','0')
    if trace not in ('0','1') or (trace=='1' and backend not in ('ss-cone-plane',)+CP_DIAGNOSTIC_BACKENDS and not (backend=='ss-cone-control' and os.environ.get('DEFORMABLE_CP_MAPPING','0') in ('observe','repair'))):raise ValueError('Cone stage trace requires ss-cone-plane or explicit CP-mapping cone-control ablation; no fallback')
    mode=os.environ.get('DEFORMABLE_KERNEL_PROFILE','0')
    states=os.environ.get('DEFORMABLE_MULTISOFT_STATES','0')
    history=os.environ.get('DEFORMABLE_MULTISOFT_HISTORY','0')
    if history not in ('0','1') or (history=='1' and states!='1'):raise ValueError('SS history requires explicit multisoft state observation')
    if backend in COMMON_BACKENDS:
        if task!='multi_soft' or states!='0' or history!='0' or mode not in ('0','1'):raise ValueError('Common SS binding requires only multi_soft; no observation composition or task fallback')
        return True
    if states not in ('0','1'):raise ValueError('Multisoft state observation must be 0 or 1')
    if states=='1' and (mode!='1' or task!='multi_soft' or backend!='native'):raise ValueError('Multisoft states require an explicit native multi_soft kernel profile')
    if mode not in ('0','1'):raise ValueError('Kernel profile must be explicitly 0 or 1')
    if mode=='1' and (backend!='native' or task not in TASKS):raise ValueError('Kernel profile requires a supported native cloth/multisoft task; no interposer composition or fallback')
    return mode=='1'
def artifacts(root,backend="native"):
    from .batched_cp_backend import BACKENDS as BATCHED,artifacts as batch_artifacts
    if backend in BATCHED:return batch_artifacts(root,backend=backend)
    from .cloth_split_backend import BACKENDS as SPLIT_BACKENDS,artifacts as split_artifacts
    if backend in SPLIT_BACKENDS:return split_artifacts(root,backend=backend)
    if os.environ.get('DEFORMABLE_CLOTH_STAGES','0')=='1':
        from .cloth_stage_backend import artifacts as cloth_artifacts
        if backend!='native':raise ValueError('Cloth stage diagnostics require native backend')
        return cloth_artifacts(root)
    if backend in CONE_BACKENDS:return cone_artifacts(root,mapping=backend in CP_DIAGNOSTIC_BACKENDS)
    if backend in COMMON_BACKENDS:return common_artifacts(root)
    root=Path(root);states=os.environ.get('DEFORMABLE_MULTISOFT_STATES','0')=='1'
    history=os.environ.get('DEFORMABLE_MULTISOFT_HISTORY','0')=='1'
    if history and not states:raise ValueError('SS history requires multisoft states')
    build=root/('results/multisoft-history-observer-build-v1' if history else 'results/multisoft-state-observer-build-v2' if states else 'results/task-kernel-profile-build-v3');manifest=build/'build.json';library=build/('libmultisoft_state_observer.so' if states else 'libtask_kernel_profile.so')
    r=json.loads(manifest.read_text());test=root/('results/multisoft-history-dispatch-selftest-v1/summary.json' if history else 'results/multisoft-profile-dispatch-selftest-v1/summary.json' if states else 'results/task-kernel-profile-selftest-v1/summary.json');t=json.loads(test.read_text())
    if r.get('status')!='compiled' or not t.get('passed'):raise ValueError('Built and tested profiler required')
    if sha(library)!=r['library_sha256']:raise ValueError('Profiler library changed')
    for name,h in {**r['input_sha256'],**t['input_sha256']}.items():
        if sha(name)!=h:raise ValueError('Profiler input changed: '+name)
    files=dict(library=library,manifest=manifest,selftest=test)
    if states:
        extra=root/('results/multisoft-history-observer-selftest-v1/summary.json' if history else 'results/multisoft-state-observer-selftest-v1/summary.json');v=json.loads(extra.read_text())
        if not v.get('passed'):raise ValueError('Owned multisoft snapshot test required')
        for name,h in v['input_sha256'].items():
            if sha(name)!=h:raise ValueError('Multisoft self-test input changed: '+name)
        files['state_selftest']=extra
    return files
class KernelProfile:
    def __init__(self,root,dest,task,backend):
        self.lib=None;self.cloth_stages=None;self.cloth_split=None;self.mapping_enabled=False;self.trace_enabled=False;self.cone_mode=CONE_BACKENDS.get(backend);self.common_mode=COMMON_BACKENDS.get(backend);self.state_window=None;self.task=task;self.path=Path(dest)/'kernel-launches.tsv'
        if not requested(task,backend):return
        files=artifacts(root,backend);self.pins={str(p):sha(p) for p in files.values()}
        if str(files['library']) not in os.environ.get('LD_PRELOAD','').split(':'):raise RuntimeError('Profiler not preloaded')
        native={line.split()[-1] for line in Path('/proc/self/maps').read_text().splitlines() if line.split() and line.split()[-1].endswith('/libPhysXGpu_64.so')}
        expected=json.loads(files['manifest'].read_text())['binary_sha256']
        if len(native)!=1 or sha(next(iter(native)))!=expected:raise RuntimeError('Unknown native GPU binary')
        self.lib=ctypes.CDLL(str(files['library']));self.lib.task_kernel_profile_start.argtypes=[ctypes.c_char_p];self.lib.task_kernel_profile_stop.restype=ctypes.c_int
        if self.lib.task_kernel_profile_start(str(self.path).encode()):self.lib=None;raise RuntimeError('Cannot start kernel profile')
        self.batched_cp=None
        if 'batch_selftest' in files:
            from .batched_cp_backend import BatchedCP
            try:self.batched_cp=BatchedCP(self.lib,files,backend,self.path.parent)
            except BaseException:
                self.lib.task_kernel_profile_stop();self.lib=None;raise
        if 'cloth_split_selftest' in files:
            from .cloth_split_backend import ClothSplit
            try:self.cloth_split=ClothSplit(self.lib,files,backend,self.path.parent)
            except BaseException:
                self.lib.task_kernel_profile_stop();self.lib=None;raise
        if 'cloth_stage_selftest' in files:
            from .cloth_stage_backend import ClothStages
            if 'cloth_trigger_selftest' in files:
                from .cloth_proximity_backend import ProximityStages as ClothStages
            try:self.cloth_stages=ClothStages(self.lib,root,self.path.parent,extended_window=64 if os.environ.get('DEFORMABLE_CLOTH_SPLIT_STAGES','0')=='1' else False)
            except BaseException:
                if self.cloth_split:self.cloth_split.stop()
                self.lib.task_kernel_profile_stop();self.lib=None;raise
        if self.common_mode is not None:
            self.lib.multisoft_common_start.argtypes=[ctypes.c_int,ctypes.c_char_p]
            self.lib.multisoft_common_stop.argtypes=[ctypes.POINTER(ctypes.c_ulonglong)]
            if self.lib.multisoft_common_start(self.common_mode,str(files['ptx']).encode()):
                self.lib.task_kernel_profile_stop();self.lib=None;raise RuntimeError('Cannot start common SS binding')
        if self.cone_mode is not None:
            self.lib.multisoft_cone_start.argtypes=[ctypes.c_int,ctypes.c_char_p,ctypes.c_char_p,ctypes.c_char_p,ctypes.c_double]
            self.lib.multisoft_cone_stop.argtypes=[ctypes.POINTER(ctypes.c_ulonglong)]
            self.cone_ledger=self.path.parent/'cone-ledger.tsv'
            if self.lib.multisoft_cone_start(self.cone_mode,str(files['cone_ptx']).encode(),str(files['split_ptx']).encode(),str(self.cone_ledger).encode(),0.11999999731779099):
                values=(ctypes.c_ulonglong*10)();self.lib.multisoft_common_stop(values);self.lib.task_kernel_profile_stop();self.lib=None
                raise RuntimeError('Cannot start cone binding')
        if 'trace_selftest' in files:
            self.lib.multisoft_cone_trace_start.argtypes=[ctypes.c_char_p]
            self.lib.multisoft_cone_trace_stop.argtypes=[ctypes.POINTER(ctypes.c_ulonglong)]
            self.trace_prefix=self.path.parent/'cone-stage'
            if self.lib.multisoft_cone_trace_start(str(self.trace_prefix).encode()):
                values=(ctypes.c_ulonglong*7)();self.lib.multisoft_cone_stop(values)
                values=(ctypes.c_ulonglong*10)();self.lib.multisoft_common_stop(values);self.lib.task_kernel_profile_stop();self.lib=None
                raise RuntimeError('Cannot start cone stage trace')
            self.trace_enabled=True
        if 'mapping_selftest' in files:
            from .cp_mapping_backend import requested as mapping_requested
            self.mapping_mode=mapping_requested(task,backend);self.mapping_ledger=self.path.parent/'cp-mapping.tsv'
            self.lib.multisoft_cp_mapping_start.argtypes=[ctypes.c_int,ctypes.c_char_p]
            self.lib.multisoft_cp_mapping_stop.argtypes=[ctypes.POINTER(ctypes.c_ulonglong)]
            if self.mapping_mode is None or self.lib.multisoft_cp_mapping_start(self.mapping_mode,str(self.mapping_ledger).encode()):
                values=(ctypes.c_ulonglong*4)();self.lib.multisoft_cone_trace_stop(values);self.trace_enabled=False
                values=(ctypes.c_ulonglong*7)();self.lib.multisoft_cone_stop(values)
                values=(ctypes.c_ulonglong*10)();self.lib.multisoft_common_stop(values);self.lib.task_kernel_profile_stop();self.lib=None
                raise RuntimeError('Cannot start CP mapping')
            self.mapping_enabled=True
        if 'state_selftest' in files:
            try:
                first=int(os.environ.get('DEFORMABLE_MULTISOFT_FIRST','120'));last=int(os.environ.get('DEFORMABLE_MULTISOFT_LAST','144'))
                if not 1<=first<=last or last-first>24:raise ValueError('Invalid bounded multisoft observation window')
                self.lib.multisoft_state_start.argtypes=[ctypes.c_char_p,ctypes.c_uint,ctypes.c_uint]
                self.lib.multisoft_state_stop.argtypes=[ctypes.POINTER(ctypes.c_ulonglong)]
                prefix=snapshot_prefix(root,self.path.parent)
                if self.lib.multisoft_state_start(str(prefix).encode(),first,last):raise RuntimeError('Cannot start multisoft snapshots')
                self.state_window=[first,last]
                for suffix in STATE_SUFFIXES:
                    (self.path.parent/('native-states'+suffix)).symlink_to(Path(str(prefix)+suffix))
            except BaseException:
                if self.state_window:
                    values=(ctypes.c_ulonglong*6)();self.lib.multisoft_state_stop(values);self.state_window=None
                self.lib.task_kernel_profile_stop();self.lib=None;raise

    def finish(self,steps):
        if self.lib is None:return None
        batch_result=self.batched_cp.stop() if getattr(self,'batched_cp',None) else None
        split_result=self.cloth_split.stop() if getattr(self,'cloth_split',None) else None
        cloth_evidence=self.cloth_stages.finish(steps) if getattr(self,'cloth_stages',None) else None
        state_evidence=None;common_evidence=None;cone_values=None;cone_status=None;trace_evidence=None
        mapping_values=None;mapping_status=None
        if self.mapping_enabled:
            mapping_values=(ctypes.c_ulonglong*6)();mapping_status=self.lib.multisoft_cp_mapping_stop(mapping_values);self.mapping_enabled=False
        if self.trace_enabled:
            values=(ctypes.c_ulonglong*4)();status=self.lib.multisoft_cone_trace_stop(values)
            files=sorted(self.path.parent.glob('cone-stage-*.json'))+sorted(self.path.parent.glob('cone-stage-*.bin'))
            trace_evidence=dict(observed_native_stages=int(values[0]),invalid_transitions=int(values[1]),snapshots=int(values[2]),errors=int(values[3]),exit_code=status,
                files_sha256={str(p):sha(p) for p in files},read_only=True,performance_acceptance=False,platform_acceptance=False)
            if mapping_values is not None:trace_evidence.update(read_only_observer=True,read_only=self.mapping_mode==0,stage_scope='native_plus_mapping_composition')
            trace_evidence['capture_valid']=status==0 and values[3]==0 and values[0]>0 and len(files)==values[2]*2
            self.trace_enabled=False
        if self.cone_mode is not None:
            cone_values=(ctypes.c_ulonglong*7)();cone_status=self.lib.multisoft_cone_stop(cone_values)
        if self.common_mode is not None:
            values=(ctypes.c_ulonglong*10)();status=self.lib.multisoft_common_stop(values)
            common_evidence=dict(mode=self.common_mode,steps=int(values[0]),solve_calls=int(values[1]),publication_calls=int(values[2]),host_errors=int(values[3]),helper_calls=int(values[4]),rejected_calls=int(values[5]),active_node_updates=int(values[6]),changed_node_updates=int(values[7]),maximum_degree=int(values[8]),changed_history_updates=int(values[9]),exit_code=status)
        if self.state_window:
            values=(ctypes.c_ulonglong*6)();status=self.lib.multisoft_state_stop(values)
            state_evidence=dict(window_steps=self.state_window,steps=int(values[0]),state_records=int(values[1]),kernel_pairs=int(values[2]),uncompressed_bytes=int(values[3]),bodies=int(values[4]),errors=int(values[5]),exit_code=status)
            state_evidence['capture_valid']=status==0 and values[5]==0 and values[0]==steps and values[1]>0 and values[1]==values[2]*4 and values[4]==2
        code=self.lib.task_kernel_profile_stop();self.lib=None
        if state_evidence:
            state_evidence['files_sha256']={str(self.path.parent/n):sha(self.path.parent/n) for n in ('native-states.jsonl','native-states.bin.gz')}
            state_evidence['storage_paths']={str(self.path.parent/n):str((self.path.parent/n).resolve()) for n in ('native-states.jsonl','native-states.bin.gz')}
        lines=self.path.read_text().splitlines();tail=re.fullmatch(r'# errors=(\d+) unsupported_launches=(\d+)',lines[-1])
        if not tail:raise RuntimeError('Incomplete profiler footer')
        counts={};rows=list(csv.DictReader(lines[:-1],delimiter='\t'))
        for row in rows:
            n=int(row['attempted']);ok=int(row['succeeded']);name=row['kernel']
            if n<ok or ok<0:raise RuntimeError('Invalid launch counters')
            counts[name]=counts.get(name,0)+ok
        checks=dict(observer_clean=code==0 and tail.groups()==('0','0'),all_launches_succeeded=all(r['attempted']==r['succeeded'] for r in rows),native_deformable_launches_present=any(n.startswith(('cloth_','sb_')) and c>0 for n,c in counts.items()))
        split_evidence=None
        if split_result is not None:
            from .cloth_split_backend import evidence as split_report
            split_evidence=split_report(*split_result,steps,counts)
            if getattr(self.cloth_split,'edge_query',False):
                from .cloth_edge_query_backend import evidence as edge_report
                if self.cloth_split.edge_query_result is None:raise RuntimeError('Missing EE query execution result')
                eq=edge_report(*self.cloth_split.edge_query_result,counts)
                split_evidence['edge_query']=eq
                split_evidence['execution_verified'] &= eq['execution_verified']
                split_evidence['repair_execution_verified'] &= eq['execution_verified']
                checks['cloth_edge_query_execution']=eq['execution_verified']
            if getattr(self.cloth_split,'owned_graph',False):
                from .cloth_owned_graph_backend import evidence as graph_report
                if self.cloth_split.owned_graph_result is None:raise RuntimeError('Missing owned graph execution result')
                ge=graph_report(*self.cloth_split.owned_graph_result,counts)
                split_evidence['owned_graph']=ge
                split_evidence['execution_verified'] &= ge['execution_verified']
                split_evidence['repair_execution_verified'] &= ge['execution_verified']
                checks['cloth_owned_graph_execution']=ge['execution_verified']
            if getattr(self.cloth_split,'history_result',None) is not None:
                from .cloth_history_backend import evidence as history_report
                he=history_report(*self.cloth_split.history_result,steps,counts,position_snapshots=2 if getattr(self.cloth_split,"transport_policy",None) is not None else 1)
                if getattr(self.cloth_split,'reentry_policy',None) is not None:
                    from .cloth_reentry_runtime_backend import policy_evidence
                    if self.cloth_split.reentry_policy_result is None:raise RuntimeError('Missing device initialization policy report')
                    pe=policy_evidence(self.cloth_split.reentry_policy,*self.cloth_split.reentry_policy_result)
                    he['initialization_policy']=pe;he['execution_verified'] &= pe['execution_verified']
                    checks['cloth_initialization_policy']=pe['execution_verified']
                if getattr(self.cloth_split,'history_stream_result',None) is not None:
                    from .cloth_history_event_backend import stream_evidence
                    se=stream_evidence(*self.cloth_split.history_stream_result,steps,counts)
                    he['stream_order']=se;he['execution_verified'] &= se['execution_verified']
                    checks['cloth_history_stream_order']=se['execution_verified']
                if getattr(self.cloth_split,'history_failure_result',None) is not None:
                    from .cloth_history_diagnostic_backend import failure_evidence
                    fe=failure_evidence(*self.cloth_split.history_failure_result)
                    he['failure_diagnostics']=fe;checks['cloth_history_failure_report']=fe['report_complete']
                    he['execution_verified'] &= fe['report_complete'] and fe['first_rejection'] is None and fe['first_decode_error'] is None
                if getattr(self.cloth_split,'history_birth_result',None) is not None:
                    from .cloth_history_birth_backend import birth_evidence
                    be=birth_evidence(*self.cloth_split.history_birth_result)
                    he['feature_birth']=be;checks['cloth_history_feature_birth']=be['execution_verified']
                    he['execution_verified'] &= be['execution_verified']
                if getattr(self.cloth_split,'history_finite_result',None) is not None:
                    from .cloth_finite_history_backend import finite_evidence
                    if getattr(self.cloth_split,'finite_initialization_result',None) is not None:
                        from .cloth_surface_reentry_backend import initialization_evidence,finite_budget_evidence
                        initialization=initialization_evidence(*self.cloth_split.finite_initialization_result)
                        fe=finite_budget_evidence(*self.cloth_split.history_finite_result,he.get('feature_birth',{}),initialization)
                    else:
                        fe=finite_evidence(*self.cloth_split.history_finite_result,he.get('feature_birth',{}))
                    he['finite_birth']=fe;checks['cloth_history_finite_birth']=fe['execution_verified']
                    he['execution_verified'] &= fe['execution_verified']
                if getattr(self.cloth_split,'inner_trace_result',None) is not None:
                    from .cloth_inner_history_backend import trace_evidence as inner_trace_evidence
                    te=inner_trace_evidence(self.cloth_split.inner_trace_path,*self.cloth_split.inner_trace_result,steps,split_evidence['ee_calls'])
                    he['inner_trace']=te;checks['cloth_inner_history_trace']=te['execution_verified']
                    he['execution_verified'] &= te['execution_verified']
                if getattr(self.cloth_split,'failure_ring_result',None) is not None:
                    from .cloth_failure_ring_backend import ring_evidence
                    ring_result=ring_evidence(str(self.cloth_split.inner_trace_path)+'.ring.bin',*self.cloth_split.failure_ring_result,steps,split_evidence['ee_calls'],he['failure_diagnostics'])
                    he['failure_ring']=ring_result;checks['cloth_failure_ring']=ring_result['execution_verified']
                    he['execution_verified'] &= ring_result['execution_verified']
                if getattr(self.cloth_split,'surface_release_result',None) is not None:
                    from .cloth_surface_release_backend import release_evidence
                    surface_result=release_evidence(*self.cloth_split.surface_release_result)
                    he['surface_release']=surface_result;checks['cloth_surface_release']=surface_result['execution_verified']
                    he['execution_verified'] &= surface_result['execution_verified']
                if getattr(self.cloth_split,'finite_initialization_result',None) is not None:
                    from .cloth_surface_reentry_backend import initialization_evidence
                    initialization_result=initialization_evidence(*self.cloth_split.finite_initialization_result)
                    if getattr(self.cloth_split,'reentry_policy',None) is not None:
                        from .cloth_reentry_runtime_backend import initialization_evidence as runtime_initialization_evidence
                        initialization_result=runtime_initialization_evidence(self.cloth_split.reentry_policy,*self.cloth_split.finite_initialization_result)
                    he['finite_initialization']=initialization_result;checks['cloth_finite_initialization']=initialization_result['execution_verified']
                    he['execution_verified'] &= initialization_result['execution_verified']
                if getattr(self.cloth_split,'transport_policy',None) is not None:
                    from .cloth_history_transport_backend import evidence as transport_evidence
                    if self.cloth_split.transport_result is None:raise RuntimeError('Missing retained-side transport counters')
                    tr=transport_evidence(self.cloth_split.transport_policy,*self.cloth_split.transport_result,split_evidence['ee_calls'],split_evidence['nodes'])
                    he['retained_side_transport']=tr;checks['cloth_retained_side_transport']=tr['execution_verified']
                    he['execution_verified'] &= tr['execution_verified']
                split_evidence['history']=he;checks['cloth_history_execution']=he['execution_verified']
                split_evidence['execution_verified'] &= he['execution_verified']
                split_evidence['repair_execution_verified'] &= he['execution_verified']
            if cloth_evidence:split_evidence['intrusive_stage_synchronization']=bool(cloth_evidence['intrusive_synchronization'])
            checks['cloth_split_execution']=split_evidence['execution_verified']
        if state_evidence:checks['multisoft_capture_valid']=state_evidence['capture_valid']
        if cloth_evidence:
            if 'trigger_monitor_valid' in cloth_evidence:
                checks['cloth_trigger_monitor_valid']=cloth_evidence['trigger_monitor_valid']
                checks['cloth_trigger_diagnostic_valid']=cloth_evidence['diagnostic_valid']
                if cloth_evidence['triggered']:checks['cloth_stage_capture_valid']=cloth_evidence['capture_valid']
            else:checks['cloth_stage_capture_valid']=cloth_evidence['capture_valid']
        if common_evidence:
            c=common_evidence;changed=(c['changed_node_updates']>0 if self.common_mode else c['changed_node_updates']==0) and (c['changed_history_updates']>0 if self.common_mode==1 else c['changed_history_updates']==0)
            c['execution_verified']=c['exit_code']==0 and c['host_errors']==0 and c['rejected_calls']==0 and c['steps']==steps and c['solve_calls']>0 and c['solve_calls']==c['publication_calls']==c['helper_calls']==counts.get('sb_solveOutputSSDeltaVLaunchTGS',0) and changed
            c['intrusive_synchronization']=True;c['physical_improvement_accepted']=False;c['performance_accepted']=False
            checks['common_ss_execution']=c['execution_verified']
        if trace_evidence:checks['cone_trace_valid']=trace_evidence['capture_valid']
        cone_evidence=None
        if cone_values is not None:
            cone_evidence=cone_evidence_report(self.cone_ledger,self.cone_mode,cone_values,cone_status,counts.get('sb_gm_stepSoftbodyLaunch',0))
            checks['cone_execution']=cone_evidence['execution_verified']
        mapping_evidence=None
        if mapping_values is not None:
            from .cp_mapping_backend import evidence as mapping_report
            mapping_evidence=mapping_report(self.mapping_ledger,self.mapping_mode,mapping_values,mapping_status,counts.get('sb_gm_cp_averageVertsLaunch',0))
            checks['cp_mapping_execution']=mapping_evidence['execution_verified']
        batch_evidence=None
        if batch_result is not None:
            from .batched_cp_backend import evidence as batch_report
            batch_evidence=batch_report(*batch_result,steps,counts);checks['batched_cp_execution']=batch_evidence['execution_verified']
            if getattr(self.batched_cp,'geometry_result',None) is not None:
                from .batched_geometry_backend import evidence as geometry_report
                geometry_evidence=geometry_report(*self.batched_cp.geometry_result,counts)
                batch_evidence['geometry']=geometry_evidence;checks['batched_geometry_capture']=geometry_evidence['capture_valid']
        result=dict(passed=all(checks.values()),checks=checks,task=self.task,completed_steps=steps,kernel_counts=counts,
            source_sha256=self.pins,raw_sha256=sha(self.path),native_state_modified=self.common_mode in (1,2) or bool(mapping_evidence and self.mapping_mode==1 and mapping_evidence['publication_calls']>0),active_contact_rows_verified=False,
            performance_acceptance=False,repair_execution_verified=False,platform_acceptance=False)
        if batch_evidence:
            result['batched_cp']=batch_evidence
            result['repair_execution_verified']=batch_evidence['repair_execution_verified']
            result['native_state_modified']=batch_evidence['repair_execution_verified']
        if split_evidence:
            result['cloth_split']=split_evidence
            result['repair_execution_verified']=split_evidence['repair_execution_verified']
            result['native_state_modified']=split_evidence['repair_execution_verified']
        if common_evidence:
            result['common_ss']=common_evidence
            result['repair_execution_verified']=self.common_mode==1 and common_evidence['execution_verified']
        if mapping_evidence:
            result['cp_mapping']=mapping_evidence
            if self.mapping_mode==1:result['repair_execution_verified']=result['repair_execution_verified'] or mapping_evidence['execution_verified']
        if trace_evidence:result['cone_trace']=trace_evidence
        if cone_evidence:result['cone_guard']=cone_evidence
        if state_evidence:result['multisoft_states']=state_evidence
        if cloth_evidence:
            result['cloth_stages']=cloth_evidence
            result['intrusive_stage_synchronization']=True
            if split_evidence:split_evidence['intrusive_stage_synchronization']=True
        (self.path.parent/'kernel-profile.json').write_text(json.dumps(result,indent=2)+'\n');return result
