"""Synchronous image observer client; IPC/serialization/wait belong in feedback timing."""
from dataclasses import asdict,is_dataclass
from pathlib import Path
import hashlib,io,os,queue,subprocess,sys,threading,time,signal
from observer_wire import VERSION,read_frame,write_frame

class ObserverWorkerError(RuntimeError):pass

class ObserverWorker:
    def __init__(self,*,source_files,numpy_module,first_sequence=0,timeout_s=20.,startup_timeout_s=30.):
        self.np=numpy_module;self.timeout_s=float(timeout_s);self.next_sequence=first_sequence;self.closed=False
        self.slow_samples=[];self.events=[];self.stderr_tail=bytearray();self._replies=queue.Queue();self._writes=queue.Queue();self._lock=threading.Lock()
        env=os.environ.copy()
        for key in ('LD_PRELOAD','LD_AUDIT','PYTHONSTARTUP','PYTHONINSPECT','PYTHONPATH','PYTHONHOME'):env.pop(key,None)
        env.update(PYTHONDONTWRITEBYTECODE='1',PYTHONUNBUFFERED='1',CUDA_VISIBLE_DEVICES='',NVIDIA_VISIBLE_DEVICES='none',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',NUMEXPR_NUM_THREADS='1')
        self.process=subprocess.Popen([sys.executable,'-B','-u',str(Path(__file__).with_name('observer_worker.py'))],stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.PIPE,env=env,bufsize=0)
        self._threads=[threading.Thread(target=self._read,daemon=True),threading.Thread(target=self._write,daemon=True),threading.Thread(target=self._stderr,daemon=True)]
        for t in self._threads:t.start()
        source_files={k:str(Path(v).resolve())for k,v in source_files.items()}
        expected={k:hashlib.sha256(Path(v).read_bytes()).hexdigest()for k,v in source_files.items()}
        try:
            self.handshake=self._exchange(dict(op='initialize',version=VERSION,numpy_file=str(Path(numpy_module.__file__).resolve()),source_files=source_files,source_sha256=expected,first_sequence=first_sequence),b'',startup_timeout_s)
            if self.handshake.get('op')!='ready'or self.handshake.get('version')!=VERSION or self.handshake.get('first_sequence')!=first_sequence or self.handshake.get('ld_preload_present')or self.handshake.get('forbidden_modules'):raise ObserverWorkerError('worker handshake mismatch')
            if self.handshake['numpy_version']!=numpy_module.__version__ or Path(self.handshake['numpy_file']).resolve()!=Path(numpy_module.__file__).resolve():raise ObserverWorkerError('worker NumPy runtime mismatch')
            if Path(self.handshake['executable']).resolve()!=Path(sys.executable).resolve():raise ObserverWorkerError('worker interpreter mismatch')
            for key,path in source_files.items():
                if self.handshake['actual_project_imports'].get(path)!=expected[key]:raise ObserverWorkerError('worker source inventory mismatch')
        except BaseException:self.abort();raise

    def _read(self):
        try:
            while True:self._replies.put(read_frame(self.process.stdout))
        except BaseException as e:self._replies.put(e)
    def _write(self):
        try:
            while True:
                item=self._writes.get()
                if item is None:return
                write_frame(self.process.stdin,*item)
        except BaseException as e:self._replies.put(e)
    def _stderr(self):
        try:
            while True:
                b=self.process.stderr.read(4096)
                if not b:return
                self.stderr_tail.extend(b)
                if len(self.stderr_tail)>1048576:del self.stderr_tail[:-1048576]
        except (OSError,ValueError):pass
    def _exchange(self,meta,data,timeout):
        self._writes.put((meta,data))
        deadline=time.monotonic()+timeout
        while True:
            remaining=deadline-time.monotonic()
            if remaining<=0:
                self.abort();raise ObserverWorkerError('observer worker timeout')
            try:
                r=self._replies.get(timeout=min(.25,remaining) if meta.get('op')=='observe' else remaining)
                break
            except queue.Empty:
                if meta.get('op')=='observe' and self.process.poll() is None and time.monotonic()<deadline:
                    sample=dict(unix=time.time(),sequence=meta['sequence'],worker_pid=self.process.pid,kind='SIGUSR1_stack_after_250ms_no_response')
                    try:os.kill(self.process.pid,signal.SIGUSR1);sample['sent']=True
                    except ProcessLookupError:sample['sent']=False
                    self.slow_samples.append(sample)
        if isinstance(r,BaseException):
            self.abort();raise ObserverWorkerError('observer worker exited or pipe failed: '+repr(r)+'; stderr: '+bytes(self.stderr_tail[-8192:]).decode('utf-8',errors='replace'))
        result,payload=r
        if payload or result.get('op')=='error':
            self.abort();raise ObserverWorkerError('observer worker rejected request: '+str(result))
        return result

    def observe(self,*,mode,depth,calibration,roi,sample_time_s,sequence,rgb=None):
        # Only image data and fixed camera/workspace metadata can cross this API.
        with self._lock:
            if self.closed:raise ObserverWorkerError('observer worker closed')
            if mode not in ('depth','color')or type(sequence)is not int or sequence!=self.next_sequence:raise ValueError('mode/sequence rejected')
            if mode=='color'and rgb is None:raise ValueError('color observer requires RGB')
            if mode=='depth'and rgb is not None:raise ValueError('depth observer cannot receive RGB')
            start=time.perf_counter();buf=io.BytesIO();arrays={'depth':depth}
            if mode=='color':arrays['rgb']=rgb
            self.np.savez(buf,**arrays);payload=buf.getvalue();serialization=time.perf_counter()-start
            meta=dict(op='observe',version=VERSION,mode=mode,calibration=asdict(calibration)if is_dataclass(calibration)else calibration,roi=asdict(roi)if is_dataclass(roi)else roi,sample_time_s=float(sample_time_s),sequence=sequence)
            result=self._exchange(meta,payload,self.timeout_s)
            if result.get('op')!='observation'or result.get('version')!=VERSION or result.get('sequence')!=sequence:
                self.abort();raise ObserverWorkerError('observer reply identity mismatch')
            self.next_sequence+=1
            self.events.append(dict(sequence=sequence,mode=mode,client_total_wall_s=time.perf_counter()-start,serialization_wall_s=serialization,payload_bytes=len(payload),worker_timing=result['worker_timing'],component_calls=result['component_calls'],component_calls_cumulative=result['component_calls_cumulative']))
            return result['result']

    def close(self):
        with self._lock:
            if self.closed:return
            try:
                reply=self._exchange(dict(op='close',version=VERSION),b'',min(self.timeout_s,5.))
                if reply.get('op')!='closed'or reply.get('next_sequence')!=self.next_sequence:raise ObserverWorkerError('worker close acknowledgement mismatch')
                self.process.wait(timeout=5.)
                if self.process.returncode!=0:raise ObserverWorkerError('worker nonzero exit')
            finally:self.abort()
    def abort(self):
        self.closed=True
        if self.process.poll()is None:
            self.process.terminate()
            try:self.process.wait(timeout=3.)
            except subprocess.TimeoutExpired:self.process.kill();self.process.wait(timeout=3.)
        self._writes.put(None)
        # Process is dead before closing pipes: release any blocked background IO.
        for s in (self.process.stdin,self.process.stdout,self.process.stderr):
            try:s.close()
            except (OSError,ValueError):pass
        for t in self._threads:
            if t is not threading.current_thread():t.join(timeout=1.)
    def __enter__(self):return self
    def __exit__(self,*exc):self.close()


