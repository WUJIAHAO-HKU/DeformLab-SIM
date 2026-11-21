"""Explicit single-cloth split-contact trial; execution is separate from acceptance."""
import ctypes,json,os
from pathlib import Path
from .cloth_stage_backend import sha
BACKENDS={'cloth-split-control':0,'cloth-split-contact':1,'cloth-split-refresh-control':0,'cloth-split-refresh-contact':1,'cloth-split-refresh-fused':1,'cloth-split-envelope-control':0,'cloth-split-envelope-contact':1,'cloth-split-envelope-refresh-control':0,'cloth-split-envelope-refresh-contact':1,'cloth-split-envelope-refresh16-control':0,'cloth-split-envelope-refresh16-contact':1}
EDGE_QUERY_BACKEND='cloth-split-envelope-refresh16-edge-query-contact'
GRAPH_BACKEND='cloth-split-envelope-refresh16-edge-query-graph-contact'
REDUCED_BACKEND='cloth-split-envelope-refresh16-edge-query-graph-reduced-contact'
COMPACT_BACKEND='cloth-split-envelope-refresh16-edge-query-graph-compact-contact'
from .compact_cloth_observer_backend import CONTROL as COMPACT_OBSERVE_CONTROL,BACKEND as COMPACT_OBSERVE_BACKEND,BACKENDS as COMPACT_OBSERVE_BACKENDS
from .cloth_surface_release_backend import BACKEND as HISTORY_SURFACE_BACKEND
from .cloth_surface_reentry_backend import BACKEND as HISTORY_REENTRY_BACKEND
from .cloth_reentry_runtime_backend import BACKENDS as REENTRY_RUNTIME_BACKENDS,POLICIES as REENTRY_RUNTIME_POLICIES
from .cloth_history_transport_backend import BACKENDS as TRANSPORT_BACKENDS,POLICIES as TRANSPORT_POLICIES
from .cloth_failure_ring_backend import BACKEND as HISTORY_RING_BACKEND
from .cloth_inner_history_backend import BACKEND as HISTORY_INNER_BACKEND
from .cloth_finite_history_backend import CONTROL as HISTORY_FINITE_CONTROL,BACKEND as HISTORY_FINITE_BACKEND,BACKENDS as HISTORY_FINITE_BACKENDS
from .cloth_history_birth_backend import CONTROL as HISTORY_BIRTH_CONTROL,BACKEND as HISTORY_BIRTH_BACKEND,BACKENDS as HISTORY_BIRTH_BACKENDS
from .cloth_history_diagnostic_backend import BACKEND as HISTORY_DIAGNOSTIC_BACKEND
from .cloth_history_event_backend import CONTROL as HISTORY_EVENT_CONTROL,BACKEND as HISTORY_EVENT_BACKEND,BACKENDS as HISTORY_EVENT_BACKENDS
from .cloth_history_backend import CONTROL as HISTORY_CONTROL,BACKEND as HISTORY_BACKEND,BACKENDS as HISTORY_BACKENDS
from .cloth_proximity_backend import CONTROL as PROXIMITY_CONTROL,BACKEND as PROXIMITY_BACKEND,BACKENDS as PROXIMITY_BACKENDS
BACKENDS.update({HISTORY_REENTRY_BACKEND:1,HISTORY_SURFACE_BACKEND:1,HISTORY_RING_BACKEND:1,HISTORY_INNER_BACKEND:1,HISTORY_FINITE_CONTROL:0,HISTORY_FINITE_BACKEND:1,HISTORY_BIRTH_CONTROL:0,HISTORY_BIRTH_BACKEND:1,HISTORY_DIAGNOSTIC_BACKEND:1,HISTORY_EVENT_CONTROL:0,HISTORY_EVENT_BACKEND:1,HISTORY_CONTROL:0,HISTORY_BACKEND:1,COMPACT_OBSERVE_CONTROL:0,COMPACT_OBSERVE_BACKEND:1,PROXIMITY_CONTROL:0,PROXIMITY_BACKEND:1})
BACKENDS.update(REENTRY_RUNTIME_BACKENDS)
BACKENDS.update(TRANSPORT_BACKENDS)
BACKENDS[COMPACT_BACKEND]=1
BACKENDS[REDUCED_BACKEND]=1
BACKENDS[GRAPH_BACKEND]=1
BACKENDS[EDGE_QUERY_BACKEND]=1
REFRESH16_BACKENDS=(*TRANSPORT_BACKENDS,*REENTRY_RUNTIME_BACKENDS,'cloth-split-envelope-refresh16-control','cloth-split-envelope-refresh16-contact',EDGE_QUERY_BACKEND,GRAPH_BACKEND,REDUCED_BACKEND,COMPACT_BACKEND,*COMPACT_OBSERVE_BACKENDS,*PROXIMITY_BACKENDS,*HISTORY_BACKENDS,*HISTORY_EVENT_BACKENDS,HISTORY_DIAGNOSTIC_BACKEND,*HISTORY_BIRTH_BACKENDS,*HISTORY_FINITE_BACKENDS,HISTORY_INNER_BACKEND,HISTORY_RING_BACKEND,HISTORY_SURFACE_BACKEND,HISTORY_REENTRY_BACKEND)
ENVELOPE_BACKENDS=('cloth-split-envelope-control','cloth-split-envelope-contact','cloth-split-envelope-refresh-control','cloth-split-envelope-refresh-contact')
REFRESH_BACKENDS=('cloth-split-refresh-control','cloth-split-refresh-contact','cloth-split-refresh-fused','cloth-split-envelope-refresh-control','cloth-split-envelope-refresh-contact')

def requested(task,backend):
    observe=os.environ.get('DEFORMABLE_CLOTH_SPLIT_STAGES','0')
    trigger=os.environ.get('DEFORMABLE_CLOTH_TRIGGER_CONFIG','')
    if bool(trigger)!=(backend in PROXIMITY_BACKENDS):raise ValueError('Explicit trigger route and config must match')
    if backend in (*COMPACT_OBSERVE_BACKENDS,*PROXIMITY_BACKENDS) and (observe!='1' or os.environ.get('DEFORMABLE_CLOTH_SELF_ROWS','0')!='1'):raise ValueError('Explicit compact diagnostic requires stage and self-row capture; no silent fallback')
    if backend in (*TRANSPORT_BACKENDS,*REENTRY_RUNTIME_BACKENDS,EDGE_QUERY_BACKEND,GRAPH_BACKEND,REDUCED_BACKEND,COMPACT_BACKEND,*HISTORY_BACKENDS,*HISTORY_EVENT_BACKENDS,HISTORY_DIAGNOSTIC_BACKEND,*HISTORY_BIRTH_BACKENDS,*HISTORY_FINITE_BACKENDS,HISTORY_INNER_BACKEND,HISTORY_RING_BACKEND,HISTORY_SURFACE_BACKEND,HISTORY_REENTRY_BACKEND) and observe!='0':raise ValueError('Edge-query backend has no intrusive observer; no fallback')
    rigid=os.environ.get('DEFORMABLE_CLOTH_RIGID_ROWS','0')
    self_rows=os.environ.get('DEFORMABLE_CLOTH_SELF_ROWS','0')
    if self_rows not in ('0','1') or (self_rows=='1' and (not observe or observe=='0' or rigid!='0' or os.environ.get('DEFORMABLE_CLOTH_GENERATION','0')!='0')):raise ValueError('Self rows require exclusive split stage diagnostics')
    generation=os.environ.get('DEFORMABLE_CLOTH_GENERATION','0')
    if generation not in ('0','1') or (generation=='1' and rigid!='1'):raise ValueError('Generation capture requires explicit rigid rows')
    if rigid not in ('0','1') or (rigid=='1' and observe!='1'):raise ValueError('Rigid rows require explicit split stage diagnostics')
    if observe not in ('0','1') or (observe=='1' and (backend not in BACKENDS or (task!='cloth_grip' and not (task=='cloth_fold' and self_rows=='1')) or backend=='cloth-split-refresh-fused')):raise ValueError('Split stage diagnostics require cloth_grip or explicit cloth_fold self rows')
    if backend not in BACKENDS:return False
    if task not in ('cloth_grip','cloth_fold'):raise ValueError('Cloth split supports one cloth in grip/fold only; no fallback')
    flags=('DEFORMABLE_CLOTH_BENCH_WINDOW','DEFORMABLE_CLOTH_STAGES','DEFORMABLE_CLOTH_CONTACTS','DEFORMABLE_MULTISOFT_STATES','DEFORMABLE_MULTISOFT_HISTORY','DEFORMABLE_CONE_TRACE','DEFORMABLE_CP_MAPPING')
    if any(os.environ.get(k,'0')!='0' for k in flags):raise ValueError('Cloth split cannot compose with other observers/repairs')
    if os.environ.get('DEFORMABLE_KERNEL_PROFILE','0') not in ('0','1'):raise ValueError('Invalid kernel profile selection')
    return True

def artifacts(root,backend=None):
    if backend in TRANSPORT_BACKENDS:
        requested("cloth_grip",backend)
        from .cloth_history_transport_backend import artifacts as transport_artifacts
        return transport_artifacts(root)
    if backend in REENTRY_RUNTIME_BACKENDS:
        requested('cloth_grip',backend)
        from .cloth_reentry_runtime_backend import artifacts as runtime_artifacts
        return runtime_artifacts(root)
    if backend==HISTORY_REENTRY_BACKEND:
        if os.environ.get("DEFORMABLE_CLOTH_SPLIT_STAGES","0")!="0":raise ValueError("No intrusive finite-initialization observer")
        from .cloth_surface_reentry_backend import artifacts as reentry_artifacts
        files=reentry_artifacts(root)
        from .cloth_geometry_checkpoint import requested as geometry_requested,artifacts as geometry_artifacts
        return geometry_artifacts(root,files) if geometry_requested('cloth_grip',backend) else files
    if backend==HISTORY_SURFACE_BACKEND:
        if os.environ.get("DEFORMABLE_CLOTH_SPLIT_STAGES","0")!="0":raise ValueError("No intrusive surface-release observer")
        from .cloth_surface_release_backend import artifacts as surface_artifacts
        return surface_artifacts(root)
    if backend==HISTORY_RING_BACKEND:
        if os.environ.get("DEFORMABLE_CLOTH_SPLIT_STAGES","0")!="0":raise ValueError("No intrusive ring stage observer")
        from .cloth_failure_ring_backend import artifacts as ring_artifacts
        return ring_artifacts(root)
    if backend==HISTORY_INNER_BACKEND:
        requested("cloth_grip",backend)
        from .cloth_inner_history_backend import artifacts as inner_artifacts
        return inner_artifacts(root)
    if backend in HISTORY_FINITE_BACKENDS:
        requested("cloth_grip",backend)
        from .cloth_finite_history_backend import artifacts as finite_artifacts
        return finite_artifacts(root)
    if backend in HISTORY_BIRTH_BACKENDS:
        requested("cloth_grip",backend)
        from .cloth_history_birth_backend import artifacts as birth_artifacts
        return birth_artifacts(root)
    if backend==HISTORY_DIAGNOSTIC_BACKEND:
        requested('cloth_grip',backend)
        from .cloth_history_diagnostic_backend import artifacts as diagnostic_artifacts
        return diagnostic_artifacts(root)
    if backend in HISTORY_EVENT_BACKENDS:
        requested('cloth_grip',backend)
        from .cloth_history_event_backend import artifacts as event_artifacts
        return event_artifacts(root)
    if backend in HISTORY_BACKENDS:
        requested('cloth_grip',backend)
        from .cloth_history_backend import artifacts as history_artifacts
        return history_artifacts(root)
    if backend in PROXIMITY_BACKENDS:
        requested('cloth_grip',backend)
        from .cloth_proximity_backend import artifacts as proximity_artifacts
        return proximity_artifacts(root)
    if backend in COMPACT_OBSERVE_BACKENDS:
        if os.environ.get('DEFORMABLE_CLOTH_SPLIT_STAGES','0')!='1' or os.environ.get('DEFORMABLE_CLOTH_SELF_ROWS','0')!='1':raise ValueError('Explicit compact observation flags required')
        from .compact_cloth_observer_backend import artifacts as observer_artifacts
        return observer_artifacts(root)
    if backend==COMPACT_BACKEND:
        if os.environ.get('DEFORMABLE_CLOTH_SPLIT_STAGES','0')!='0':raise ValueError('No compact-grid intrusive observer')
        from .cloth_compact_grid_backend import artifacts as compact_artifacts
        return compact_artifacts(root)
    if backend==REDUCED_BACKEND:
        if os.environ.get('DEFORMABLE_CLOTH_SPLIT_STAGES','0')!='0':raise ValueError('No reduced-count intrusive observer')
        from .cloth_reduced_count_backend import artifacts as reduced_artifacts
        return reduced_artifacts(root)
    if backend==GRAPH_BACKEND:
        if os.environ.get('DEFORMABLE_CLOTH_SPLIT_STAGES','0')!='0':raise ValueError('No graph intrusive observer')
        from .cloth_owned_graph_backend import artifacts as graph_artifacts
        return graph_artifacts(root)
    if backend==EDGE_QUERY_BACKEND:
        if os.environ.get('DEFORMABLE_CLOTH_SPLIT_STAGES','0')!='0':raise ValueError('No edge-query intrusive observer')
        from .cloth_edge_query_backend import artifacts as edge_artifacts
        return edge_artifacts(root)
    root=Path(root);observe=os.environ.get('DEFORMABLE_CLOTH_SPLIT_STAGES','0')=='1';fused=backend=='cloth-split-refresh-fused'
    if observe and fused:raise ValueError('Fused backend has no matching intrusive observer; no fallback')
    rigid=os.environ.get('DEFORMABLE_CLOTH_RIGID_ROWS','0')
    self_rows=os.environ.get('DEFORMABLE_CLOTH_SELF_ROWS','0')
    if self_rows not in ('0','1') or (self_rows=='1' and (not observe or observe=='0' or rigid!='0' or os.environ.get('DEFORMABLE_CLOTH_GENERATION','0')!='0')):raise ValueError('Self rows require exclusive split stage diagnostics')
    generation=os.environ.get('DEFORMABLE_CLOTH_GENERATION','0')
    if generation not in ('0','1') or (generation=='1' and rigid!='1'):raise ValueError('Generation capture requires explicit rigid rows')
    if rigid not in ('0','1') or (rigid=='1' and not observe):raise ValueError('Rigid rows require explicit split stage diagnostics')
    rigid=rigid=='1'
    build=root/('results/cloth-self-rows-observer-build-v1' if self_rows=='1' else 'results/cloth-generation-observer-build-v2' if generation=='1' else 'results/cloth-rigid-rows-observer-build-v1' if rigid else 'results/cloth-split-observer-build-v2' if observe else 'results/cloth-split-backend-build-v4' if fused else 'results/cloth-split-backend-build-v3')
    files=dict(library=build/'libcloth_split_backend.so',manifest=build/'build.json',cloth_split_ptx=build/'contact.ptx',
        cloth_split_selftest=root/('results/cloth-split-observer-binding-selftest-v2/summary.json' if observe else 'results/cloth-split-binding-selftest-v4/summary.json' if fused else 'results/cloth-split-binding-selftest-v3/summary.json'),
        selftest=root/('results/cloth-split-observer-dispatch-selftest-v2/summary.json' if observe else 'results/cloth-split-dispatch-selftest-v4/summary.json' if fused else 'results/cloth-split-dispatch-selftest-v3/summary.json'),
        cache_selftest=root/('results/cloth-split-observer-cache-selftest-v2/summary.json' if observe else 'results/cloth-profile-cache-selftest-v4/summary.json' if fused else 'results/cloth-profile-cache-selftest-v1/summary.json'),
        coulomb_vt=root/'results/cloth-isolated-coulomb-vt-weighted-v1/summary.json',
        coulomb_ee=root/'results/cloth-isolated-coulomb-ee-weighted-v1/summary.json')
    if observe:files['cloth_stage_selftest']=root/'results/cloth-split-observer-window-selftest-v2/summary.json'
    if rigid:
        for key,directory in {'cloth_split_selftest':'cloth-rigid-rows-binding-selftest-v1','selftest':'cloth-rigid-rows-dispatch-selftest-v1','cache_selftest':'cloth-rigid-rows-cache-selftest-v1','cloth_stage_selftest':'cloth-rigid-rows-observer-selftest-v1'}.items():files[key]=root/'results'/directory/'summary.json'
    if generation=='1':
        for key,directory in {'cloth_split_selftest':'cloth-generation-binding-selftest-v2','selftest':'cloth-generation-dispatch-selftest-v2','cache_selftest':'cloth-generation-cache-selftest-v2','cloth_stage_selftest':'cloth-generation-rigid-capture-selftest-v2','generation_selftest':'cloth-generation-observer-selftest-v2'}.items():files[key]=root/'results'/directory/'summary.json'
    if self_rows=='1':
        for key,directory in {'cloth_split_selftest':'cloth-self-rows-binding-selftest-v1','selftest':'cloth-self-rows-dispatch-selftest-v1','cache_selftest':'cloth-self-rows-cache-selftest-v1','cloth_stage_selftest':'cloth-self-rows-observer-selftest-v1'}.items():files[key]=root/'results'/directory/'summary.json'
    manifest=json.loads(files['manifest'].read_text())
    if manifest.get('status')!='compiled' or sha(files['library'])!=manifest['library_sha256'] or sha(files['cloth_split_ptx'])!=manifest['ptx_sha256']:raise ValueError('Verified cloth split build required')
    if fused:
        expected={'cloth_split_vt_normal','cloth_split_ee_normal','cloth_split_vt_friction','cloth_split_ee_friction','cloth_split_vt_query','cloth_split_ee_query','cloth_split_bind_frozen','cloth_split_restore_degree','cloth_applyExternalDeltasLaunch'}
        unchanged=manifest.get('unchanged_contact_entries',{})
        if set(unchanged)!=expected or not all(v is True for v in unchanged.values()):raise ValueError('Fused backend changed contact entries')
    elif sha(manifest['gpu_kernels_byte_identical_to'])!=manifest['ptx_sha256']:raise ValueError('Cached backend must preserve GPU kernel bytes')
    pins=dict(manifest['input_sha256'])
    for key in ('cloth_split_selftest','selftest','cache_selftest')+(('cloth_stage_selftest',) if observe else ())+(('generation_selftest',) if generation=='1' else ()):
        receipt=json.loads(files[key].read_text())
        if not receipt.get('passed'):raise ValueError('Passing cloth split prerequisites required')
        pins.update(receipt['input_sha256'])
    for key in ('coulomb_vt','coulomb_ee'):
        receipt=json.loads(files[key].read_text());protocol=files[key].parent/'protocol.json'
        if not receipt.get('passed') or not receipt.get('query_verified') or sha(protocol)!=receipt['protocol_sha256']:raise ValueError('Independent Coulomb reference must pass')
        pins.update(json.loads(protocol.read_text())['input_sha256']);files[key+'_protocol']=protocol
    if any(sha(p)!=h for p,h in pins.items()):raise ValueError('Cloth split input changed')
    return files

def evidence(mode,values,status,steps,counts):
    v=list(map(int,values));gpu=v[6:16]
    coverage=v[0]==steps and steps>0 and v[1]>0 and v[2]>0 and v[3]==v[1]+v[2] and v[1]==counts.get('cloth_solveClothClothDeltaVTLaunch',0) and v[2]==counts.get('cloth_solveClothClothDeltaEELaunch',0)
    execution=(gpu[:4]==[v[1],v[1],v[2],v[2]] and gpu[8:10]==[v[1],v[2]] and gpu[4]>0 and gpu[5]>0 and gpu[7]>0) if mode else gpu==[0]*10
    verified=status==0 and v[4]==0 and gpu[6]==0 and v[5]==mode and v[16]>=3 and coverage and execution
    return dict(mode=mode,steps=v[0],vt_calls=v[1],ee_calls=v[2],publication_calls=v[3],host_errors=v[4],gpu_counts=gpu,
        nodes=v[16],scratch_bytes=v[17],exit_code=status,execution_verified=bool(verified),repair_execution_verified=bool(mode and verified),
        scope='Single-cloth VT/EE normal/friction split; rigid-cloth contacts and material response remain native.',
        intrusive_stage_synchronization=False,physical_improvement_accepted=False,performance_accepted=False,platform_acceptance=False)

class ClothSplit:
    def __init__(self,lib,files,backend,output_dir=None):
        self.transport_policy=TRANSPORT_POLICIES.get(backend);self.transport_result=None
        self.reentry_policy=1 if self.transport_policy is not None else REENTRY_RUNTIME_POLICIES.get(backend);self.reentry_policy_result=None
        if self.reentry_policy is not None:backend=HISTORY_REENTRY_BACKEND
        self.lib=lib;self.mode=BACKENDS[backend];self.stopped=False;self.history=backend in (HISTORY_BACKEND,HISTORY_EVENT_BACKEND,HISTORY_DIAGNOSTIC_BACKEND,HISTORY_BIRTH_BACKEND,HISTORY_FINITE_BACKEND,HISTORY_INNER_BACKEND,HISTORY_RING_BACKEND,HISTORY_SURFACE_BACKEND,HISTORY_REENTRY_BACKEND);self.history_events=backend in (HISTORY_EVENT_BACKEND,HISTORY_DIAGNOSTIC_BACKEND,HISTORY_BIRTH_BACKEND,HISTORY_FINITE_BACKEND,HISTORY_INNER_BACKEND,HISTORY_RING_BACKEND,HISTORY_SURFACE_BACKEND,HISTORY_REENTRY_BACKEND);self.inner_trace=backend in (HISTORY_INNER_BACKEND,HISTORY_RING_BACKEND,HISTORY_SURFACE_BACKEND,HISTORY_REENTRY_BACKEND);self.failure_ring=backend in (HISTORY_RING_BACKEND,HISTORY_SURFACE_BACKEND,HISTORY_REENTRY_BACKEND);self.surface_release=backend in (HISTORY_SURFACE_BACKEND,HISTORY_REENTRY_BACKEND);self.finite_initialization=backend==HISTORY_REENTRY_BACKEND;self.finite_initialization_result=None;self.surface_release_result=None;self.failure_ring_result=None;self.inner_trace_result=None;self.inner_trace_path=None;self.history_finite=backend in (HISTORY_FINITE_BACKEND,HISTORY_INNER_BACKEND,HISTORY_RING_BACKEND,HISTORY_SURFACE_BACKEND,HISTORY_REENTRY_BACKEND);self.history_finite_result=None;self.history_birth=backend in (HISTORY_BIRTH_BACKEND,HISTORY_FINITE_BACKEND,HISTORY_INNER_BACKEND,HISTORY_RING_BACKEND,HISTORY_SURFACE_BACKEND,HISTORY_REENTRY_BACKEND);self.history_birth_result=None;self.history_diagnostic=backend in (HISTORY_DIAGNOSTIC_BACKEND,HISTORY_BIRTH_BACKEND,HISTORY_FINITE_BACKEND,HISTORY_INNER_BACKEND,HISTORY_RING_BACKEND,HISTORY_SURFACE_BACKEND,HISTORY_REENTRY_BACKEND);self.history_failure_result=None;self.history_stream_result=None;self.history_result=None;self.edge_query=backend in (HISTORY_REENTRY_BACKEND,HISTORY_SURFACE_BACKEND,HISTORY_RING_BACKEND,HISTORY_INNER_BACKEND,HISTORY_FINITE_BACKEND,HISTORY_BIRTH_BACKEND,HISTORY_DIAGNOSTIC_BACKEND,HISTORY_EVENT_BACKEND,HISTORY_BACKEND,EDGE_QUERY_BACKEND,GRAPH_BACKEND,REDUCED_BACKEND,COMPACT_BACKEND,COMPACT_OBSERVE_BACKEND,PROXIMITY_BACKEND);self.edge_query_result=None;self.owned_graph=backend in (GRAPH_BACKEND,REDUCED_BACKEND,COMPACT_BACKEND,COMPACT_OBSERVE_BACKEND,PROXIMITY_BACKEND);self.owned_graph_result=None
        if self.reentry_policy is not None:self.inner_trace=False;self.failure_ring=False
        lib.cloth_split_start.argtypes=[ctypes.c_int,ctypes.c_char_p];lib.cloth_split_stop.argtypes=[ctypes.POINTER(ctypes.c_ulonglong)]
        if lib.cloth_split_start(self.mode,str(files['cloth_split_ptx']).encode()):raise RuntimeError('Cannot activate cloth split backend')
        if self.edge_query:
            lib.cloth_edge_query_report.argtypes=[ctypes.POINTER(ctypes.c_ulonglong)]
            if lib.cloth_edge_query_start():
                values=(ctypes.c_ulonglong*18)();lib.cloth_split_stop(values);self.stopped=True
                raise RuntimeError('Cannot explicitly activate corrected EE query')
        if self.history:
            lib.cloth_history_report.argtypes=[ctypes.POINTER(ctypes.c_ulonglong)]
            if lib.cloth_history_start():
                values=(ctypes.c_ulonglong*18)();lib.cloth_split_stop(values);self.stopped=True
                raise RuntimeError('Cannot explicitly activate persistent EE history')
        if self.reentry_policy is not None:
            lib.cloth_history_reentry_policy_start.argtypes=[ctypes.c_uint]
            if lib.cloth_history_reentry_policy_start(self.reentry_policy):
                values=(ctypes.c_ulonglong*18)();lib.cloth_split_stop(values);self.stopped=True
                raise RuntimeError('Cannot explicitly select initialization policy')
        if self.transport_policy is not None:
            lib.cloth_history_transport_policy_start.argtypes=[ctypes.c_uint]
            if lib.cloth_history_transport_policy_start(self.transport_policy):
                values=(ctypes.c_ulonglong*18)();lib.cloth_split_stop(values);self.stopped=True
                raise RuntimeError("Cannot explicitly select retained-side transport policy")
        if self.inner_trace:
            from .cloth_inner_history_backend import KEYS
            if output_dir is None:raise ValueError('Explicit inner trace output directory required')
            self.inner_trace_path=Path(output_dir)/'cloth-inner-history.bin'
            keys=(ctypes.c_ulonglong*2)(*KEYS)
            lib.cloth_history_trace_start.argtypes=[ctypes.c_char_p,ctypes.POINTER(ctypes.c_ulonglong),ctypes.c_uint]
            if lib.cloth_history_trace_start(str(self.inner_trace_path).encode(),keys,2):
                values=(ctypes.c_ulonglong*18)();lib.cloth_split_stop(values);self.stopped=True
                raise RuntimeError('Cannot explicitly activate inner history trace')
        if self.owned_graph:
            lib.cloth_owned_graph_report.argtypes=[ctypes.POINTER(ctypes.c_ulonglong)]
            if lib.cloth_owned_graph_start():
                values=(ctypes.c_ulonglong*18)();lib.cloth_split_stop(values);self.stopped=True
                raise RuntimeError('Cannot explicitly activate owned graph')
    def stop(self):
        if self.stopped:raise RuntimeError('Cloth split already stopped')
        values=(ctypes.c_ulonglong*18)();code=self.lib.cloth_split_stop(values);self.stopped=True
        if self.edge_query:
            q=(ctypes.c_ulonglong*6)();qc=self.lib.cloth_edge_query_report(q);self.edge_query_result=(list(q),qc)
        if self.owned_graph:
            g=(ctypes.c_ulonglong*5)();gc=self.lib.cloth_owned_graph_report(g);self.owned_graph_result=(list(g),gc)
        if self.history:
            h=(ctypes.c_ulonglong*16)();hc=self.lib.cloth_history_report(h);self.history_result=(list(h),hc)
        if self.history_events:
            self.lib.cloth_history_stream_report.argtypes=[ctypes.POINTER(ctypes.c_ulonglong)]
            e=(ctypes.c_ulonglong*5)();ec=self.lib.cloth_history_stream_report(e);self.history_stream_result=(list(e),ec)
        if self.history_diagnostic:
            self.lib.cloth_history_failure_report.argtypes=[ctypes.POINTER(ctypes.c_uint),ctypes.POINTER(ctypes.c_uint)]
            f=(ctypes.c_uint*128)();d=(ctypes.c_uint*16)();dc=self.lib.cloth_history_failure_report(f,d);self.history_failure_result=(list(f),list(d),dc)
        if self.history_birth:
            self.lib.cloth_history_birth_report.argtypes=[ctypes.POINTER(ctypes.c_ulonglong)]
            b=(ctypes.c_ulonglong*3)();bc=self.lib.cloth_history_birth_report(b);self.history_birth_result=(list(b),bc)
        if self.history_finite:
            self.lib.cloth_history_finite_report.argtypes=[ctypes.POINTER(ctypes.c_ulonglong)]
            f=(ctypes.c_ulonglong*5)();fc=self.lib.cloth_history_finite_report(f);self.history_finite_result=(list(f),fc)
        if self.inner_trace:
            self.lib.cloth_history_trace_report.argtypes=[ctypes.POINTER(ctypes.c_ulonglong)]
            t=(ctypes.c_ulonglong*6)();tc=self.lib.cloth_history_trace_report(t);self.inner_trace_result=(list(t),tc)
        if self.failure_ring:
            self.lib.cloth_history_ring_report.argtypes=[ctypes.POINTER(ctypes.c_ulonglong)]
            r=(ctypes.c_ulonglong*9)();rc=self.lib.cloth_history_ring_report(r);self.failure_ring_result=(list(r),rc)
        if self.surface_release:
            self.lib.cloth_history_surface_release_report.argtypes=[ctypes.POINTER(ctypes.c_ulonglong)]
            r=(ctypes.c_ulonglong*3)();rc=self.lib.cloth_history_surface_release_report(r);self.surface_release_result=(list(r),rc)
        if self.finite_initialization:
            self.lib.cloth_history_reentry_report.argtypes=[ctypes.POINTER(ctypes.c_ulonglong)]
            r=(ctypes.c_ulonglong*3)();rc=self.lib.cloth_history_reentry_report(r);self.finite_initialization_result=(list(r),rc)
        if self.reentry_policy is not None:
            self.lib.cloth_history_reentry_policy_report.argtypes=[ctypes.POINTER(ctypes.c_uint)]
            r=(ctypes.c_uint*3)();rc=self.lib.cloth_history_reentry_policy_report(r);self.reentry_policy_result=(list(r),rc)
        if self.transport_policy is not None:
            self.lib.cloth_history_transport_report.argtypes=[ctypes.POINTER(ctypes.c_ulonglong)]
            self.lib.cloth_history_transport_policy_report.argtypes=[ctypes.POINTER(ctypes.c_uint)]
            r=(ctypes.c_ulonglong*10)();rc=self.lib.cloth_history_transport_report(r)
            p=(ctypes.c_uint*3)();pc=self.lib.cloth_history_transport_policy_report(p);self.transport_result=(list(r),rc,list(p),pc)
        return self.mode,list(values),code
