"""Explicit, version-locked bridge for the experimental conservative CP update."""
import ctypes,hashlib,os,sys
from pathlib import Path

class ConservativeCpTrial:
    def __init__(self,root,physx_interface,path,mode,weight_backend=None):
        self.lib=None;root=Path(root)
        self.weight_backend=weight_backend;weight_artifacts=None
        library=Path('./results/paper-method-components-build-v2/sparse/libphysx_robot_surface.so')
        if weight_backend is not None:
            from deformable_tasks.weight_backend import artifacts
            if mode!='pass-through':raise ValueError('Weight experiment requires unchanged native material mode')
            weight_artifacts=artifacts(root,weight_backend);library=weight_artifacts['library']
        if str(library) not in os.environ.get('LD_PRELOAD','').split(':'):raise RuntimeError('CP trial interposer was not preloaded')
        libraries={line.split()[-1] for line in Path('/proc/self/maps').read_text().splitlines() if line.split() and line.split()[-1].endswith('/libPhysXGpu_64.so')}
        if len(libraries)!=1:raise RuntimeError('Expected exactly one PhysX GPU binary')
        gpu=Path(next(iter(libraries)));sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
        if sha(gpu)!='05220192126a9bacedeb432eca6070be2faddc5879475b90be00bfeb815c1cc7':raise RuntimeError('Unsupported PhysX GPU version')
        sys.path.insert(0,str(root/'work/native-readback'))
        import physx_deformable_readback
        self.handle=physx_deformable_readback.simulation_buffer(physx_interface,path)
        self.lib=ctypes.CDLL(str(library));self.lib.observer_start.argtypes=[ctypes.c_char_p,ctypes.c_ulonglong,ctypes.c_ulonglong,ctypes.c_size_t]
        self.lib.observer_enable_trial.argtypes=[ctypes.c_char_p,ctypes.c_int]
        self.lib.observer_trial_count.argtypes=[ctypes.c_int];self.lib.observer_trial_count.restype=ctypes.c_ulonglong
        self.lib.observer_stop.restype=ctypes.c_uint
        h=self.handle;c=self.lib.observer_start(b'',h['device_pointer'],h['velocity_pointer'],h['node_count'])
        if c:raise RuntimeError(f'CP context binding failed: {c}')
        ptx=root/'work/kernel-observer/physx_cp_conservative_trial.ptx'
        c=self.lib.observer_enable_trial(str(ptx).encode(),{'split-velocity':2,'split-sequential':4,'split-stale':5,'pass-through':3}[mode])
        if c:self.stop();raise RuntimeError(f'CP trial start failed: {c}')
        self.evidence=dict(mode=mode,gpu_library=str(gpu),gpu_sha256=sha(gpu),hook_sha256=sha(library),ptx_sha256=sha(ptx),node_count=h['node_count'],native_settings=h['settings'],scope='Experimental host driver interposition in the fixed PhysX binary; not an upstream solver patch or universal material validation.')
        if weight_artifacts:
            self.lib.observer_weight_contact_start.argtypes=[ctypes.c_char_p]
            self.lib.observer_weight_contact_start.restype=ctypes.c_int
            self.lib.observer_weight_contact_count.restype=ctypes.c_ulonglong
            self.lib.observer_weight_contact_stats.argtypes=[ctypes.POINTER(ctypes.c_uint)]
            self.lib.observer_weight_contact_stats.restype=ctypes.c_int
            code=self.lib.observer_weight_contact_start(str(weight_artifacts['ptx']).encode())
            if code:self.stop();raise RuntimeError('Weight contact activation failed: '+str(code))
            self.evidence['weight_contact']=dict(backend=weight_backend,ptx_sha256=sha(weight_artifacts['ptx']),build_sha256=sha(weight_artifacts['manifest']),execution_verified=False,platform_acceptance=False)
            if 'query_ptx' in weight_artifacts:
                self.lib.observer_support_query_start.argtypes=[ctypes.c_char_p]
                self.lib.observer_support_query_start.restype=ctypes.c_int
                self.lib.observer_support_query_count.restype=ctypes.c_ulonglong
                self.lib.observer_support_query_stats.argtypes=[ctypes.POINTER(ctypes.c_uint)]
                self.lib.observer_support_query_stats.restype=ctypes.c_int
                code=self.lib.observer_support_query_start(str(weight_artifacts['query_ptx']).encode())
                if code:self.stop();raise RuntimeError('Support query activation failed: '+str(code))
                self.evidence['weight_contact']['support_query']=dict(ptx_sha256=sha(weight_artifacts['query_ptx']),execution_verified=False)
    def weight_evidence(self,expected_rounds):
        if self.weight_backend is None:return None
        import struct
        data=(ctypes.c_uint*8)()
        if self.lib.observer_weight_contact_stats(data):raise RuntimeError('Cannot read native weight execution statistics')
        calls=int(self.lib.observer_weight_contact_count());processed,invalid,changed=int(data[0]),int(data[1]),int(data[2])
        raw=struct.unpack('<f',struct.pack('<I',data[3]))[0];effective=struct.unpack('<f',struct.pack('<I',data[4]))[0]
        normalized=self.weight_backend!='cp-contact-weights-control'
        passed=calls==expected_rounds and processed>0 and invalid==0 and (effective<=2e-6 if normalized else changed==0 and effective==raw)
        e=self.evidence['weight_contact'];e.update(raw_statistics=[int(x) for x in data],calls=calls,expected_calls=expected_rounds,processed_active_rows=processed,invalid_rows=invalid,changed_rows=changed,maximum_raw_weight_defect=raw,maximum_effective_weight_defect=effective,execution_verified=passed,passed=passed)
        if 'support_query' in e:
            counters=(ctypes.c_uint*2)()
            if self.lib.observer_support_query_stats(counters):raise RuntimeError('Cannot read support query statistics')
            query_calls=int(self.lib.observer_support_query_count())
            positive=self.weight_backend=='cp-contact-weights-support'
            valid=query_calls==expected_rounds and counters[0]>0 and (counters[1]>0 if positive else counters[1]==0)
            e['support_query'].update(raw_statistics=[int(x) for x in counters],calls=query_calls,expected_calls=expected_rounds,total_reference_updates=int(counters[0]),additional_reference_updates=int(counters[1]),execution_verified=valid,passed=valid)
            e['passed']=e['execution_verified']=bool(passed and valid)
        return e
    def counts(self):return [int(self.lib.observer_trial_count(i)) for i in range(7)]
    def trace_start(self,path):
        Path(str(path)+'.inputs').mkdir(exist_ok=False)
        self.lib.observer_trace_start.argtypes=[ctypes.c_char_p]
        self.lib.observer_trace_step.argtypes=[ctypes.c_int]
        code=self.lib.observer_trace_start(str(path).encode())
        if code:raise RuntimeError(f'CP trace start failed: {code}')
    def trace_step(self,step):self.lib.observer_trace_step(step)
    def stop(self):
        if self.lib is None:return 0
        errors=int(self.lib.observer_stop());self.lib=None;return errors
