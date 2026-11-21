"""Persistent image-only observer subprocess. No Kit, CUDA, pickle or disk output."""
from pathlib import Path
import hashlib,importlib.util,io,json,os,sys,time,traceback,signal,faulthandler
from low_overhead import snapshot,difference
from observer_wire import VERSION,read_frame,write_frame

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def load(name,path):
    s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);sys.modules[name]=m;s.loader.exec_module(m);return m

def main():
    inp=sys.stdin.buffer;out=sys.stdout.buffer
    # Imported source diagnostics cannot corrupt binary stdout framing.
    sys.stdout=sys.stderr
    try:
        hello,payload=read_frame(inp)
        if payload or set(hello)!={'op','version','numpy_file','source_files','source_sha256','first_sequence'} or hello['op']!='initialize' or hello['version']!=VERSION:raise ValueError('invalid initialize message')
        npfile=Path(hello['numpy_file']).resolve();sys.path.insert(0,str(npfile.parent.parent))
        import numpy as np
        if Path(np.__file__).resolve()!=npfile:raise ValueError('NumPy import path differs from caller')
        from vectorized_geometry import VectorizedReusedDepthObserver
        sources=hello['source_files'];expected=hello['source_sha256']
        if set(sources)!={'observers','side_push_config','side_push_observers','component_flat'} or set(expected)!=set(sources):raise ValueError('invalid source inventory')
        for name,path in sources.items():
            if sha(path)!=expected[name]:raise ValueError('source digest mismatch: '+name)
        oracle=load('observers',sources['observers']);config=load('side_push_config',sources['side_push_config']);side=load('side_push_observers',sources['side_push_observers']);flat=load('component_flat',sources['component_flat'])
        faulthandler.register(signal.SIGUSR1,file=sys.stderr,all_threads=False,chain=False)
        geometry_observer=VectorizedReusedDepthObserver(side)
        side.observe_depth=geometry_observer
        component_count=0
        def counted_component(mask):
            nonlocal component_count
            component_count+=1
            return flat._largest_component(mask)
        side._largest_component=counted_component
        source_hashes={str(Path(path).resolve()):sha(path)for path in sources.values()}
        for path in (Path(__file__),Path(__file__).with_name('observer_wire.py'),Path(__file__).with_name('low_overhead.py'),Path(__file__).with_name('reused_geometry.py'),Path(__file__).with_name('vectorized_geometry.py')):source_hashes[str(path.resolve())]=sha(path)
        forbidden=[name for name in sys.modules if name=='isaacsim' or name.startswith(('isaacsim.','omni.','pxr.','warp.','torch.','carb.'))]
        if forbidden:raise RuntimeError('unexpected simulation module import')
        sequence=hello['first_sequence']
        if type(sequence)is not int or sequence<0:raise ValueError('invalid initial sequence')
        write_frame(out,dict(op='ready',version=VERSION,pid=os.getpid(),python=sys.version,executable=sys.executable,numpy_version=np.__version__,numpy_file=str(npfile),actual_project_imports=source_hashes,first_sequence=sequence,ld_preload_present=bool(os.environ.get('LD_PRELOAD')),gpu_visibility=os.environ.get('CUDA_VISIBLE_DEVICES'),thread_limits={k:os.environ.get(k) for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS')},forbidden_modules=forbidden,geometry_implementation='reused-float64-ordered-three-term-norm',trace_active=sys.gettrace() is not None,profile_active=sys.getprofile() is not None))
        while True:
            meta,payload=read_frame(inp)
            if meta=={'op':'close','version':VERSION}and not payload:
                write_frame(out,dict(op='closed',version=VERSION,next_sequence=sequence));return
            allowed={'op','version','mode','calibration','roi','sample_time_s','sequence'}
            if set(meta)!=allowed or meta['op']!='observe' or meta['version']!=VERSION:raise ValueError('invalid observation schema')
            if type(meta['sequence'])is not int or meta['sequence']!=sequence:raise ValueError('unexpected observation sequence')
            if meta['mode']not in ('depth','color'):raise ValueError('worker only accepts image observers')
            calibration=oracle.Calibration(**meta['calibration']);calibration.matrix()
            roi=oracle.NominalROI(**meta['roi']);roi.bounds()
            # Existing side observer has a fixed ROI; require agreement rather than override its semantics.
            if tuple(roi.world_min)!=tuple(side.ROI.world_min)or tuple(roi.world_max)!=tuple(side.ROI.world_max):raise ValueError('ROI differs from frozen observer')
            if not np.isfinite(meta['sample_time_s'])or meta['sample_time_s']<0:raise ValueError('invalid sample time')
            before_component_count=component_count;start=time.perf_counter();cpu=time.process_time();thread=time.thread_time()
            with np.load(io.BytesIO(payload),allow_pickle=False)as data:
                required={'depth','rgb'}if meta['mode']=='color'else {'depth'}
                if set(data.files)!=required:raise ValueError('only RGB/depth arrays are accepted')
                depth=data['depth']
                if depth.shape!=(calibration.height,calibration.width)or depth.dtype.kind not in 'fu':raise ValueError('invalid depth array')
                args=(depth,calibration)
                if meta['mode']=='color':
                    rgb=data['rgb']
                    if rgb.ndim!=3 or rgb.shape[:2]!=depth.shape or rgb.shape[2]not in (3,4)or rgb.dtype.kind not in 'fui':raise ValueError('invalid RGB array')
                    args=(rgb,depth,calibration)
                low_before=snapshot()
                obs_start=time.perf_counter();result=getattr(side,'observe_'+meta['mode'])(*args,sample_time_s=meta['sample_time_s'],sequence=sequence);obs_wall=time.perf_counter()-obs_start
                low_after=snapshot()
            write_frame(out,dict(op='observation',version=VERSION,sequence=sequence,result=result,worker_timing=dict(total_wall_s=time.perf_counter()-start,process_cpu_s=time.process_time()-cpu,thread_cpu_s=time.thread_time()-thread,observer_wall_s=obs_wall,low_overhead=difference(low_before,low_after),geometry_workspace_allocations=geometry_observer.allocations),component_calls=component_count-before_component_count,component_calls_cumulative=component_count))
            sequence+=1
    except BaseException as e:
        try:write_frame(out,dict(op='error',version=VERSION,error=type(e).__name__+': '+str(e)))
        except BaseException:pass
        traceback.print_exc(file=sys.stderr);raise

if __name__=='__main__':main()

