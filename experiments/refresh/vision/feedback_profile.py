"""Host-boundary timings; GPU work may complete at later readback boundaries."""
from contextlib import contextmanager
from io import BytesIO
import json,time,gc
from pathlib import Path
import resource

def cpu_snapshot():
    usage=resource.getrusage(resource.RUSAGE_SELF)
    result=dict(voluntary_switches=usage.ru_nvcsw,involuntary_switches=usage.ru_nivcsw)
    path=Path('/sys/fs/cgroup/cpu.stat')
    if path.exists(): result['cgroup_cpu_stat']={k:int(v) for k,v in (line.split() for line in path.read_text().splitlines())}
    return result

class FeedbackProfile:
    def __init__(self):
        self.events=[];self.current=None;self.gc_events=[];self.gc_start=None;self.active_stage=None
        self.gc_threshold=gc.get_threshold()
        gc.callbacks.append(self.gc_callback)
    def gc_callback(self,phase,info):
        if phase=='start':
            self.gc_start=(time.monotonic(),time.process_time(),self.active_stage,self.current['step'] if self.current else None)
        elif self.gc_start:
            wall,cpu,stage,step=self.gc_start
            self.gc_events.append(dict(started_monotonic=wall,wall_s=time.monotonic()-wall,cpu_s=time.process_time()-cpu,stage=stage,step=step,**info))
            self.gc_start=None
    def begin(self,step):
        self.current=dict(step=step,in_rollout=step>0,started_unix=time.time(),started_monotonic=time.monotonic(),stages={})
        self.current['cpu_before']=cpu_snapshot()
    @contextmanager
    def stage(self,name):
        wall=time.monotonic();cpu=time.process_time();thread=time.thread_time();self.active_stage=name
        try: yield
        finally:
            self.current['stages'][name]=dict(wall_s=time.monotonic()-wall,cpu_s=time.process_time()-cpu,thread_cpu_s=time.thread_time()-thread)
            self.active_stage=None
    def end(self):
        self.current['cpu_after']=cpu_snapshot()
        self.current['elapsed_s']=time.monotonic()-self.current['started_monotonic']
        self.current['unattributed_s']=self.current['elapsed_s']-sum(s['wall_s'] for s in self.current['stages'].values())
        self.events.append(self.current);self.current=None
    def archive(self,path,np,**arrays):
        with self.stage('npz_compression'):
            buffer=BytesIO();np.savez_compressed(buffer,**arrays)
        with self.stage('npz_write'):
            with path.open('wb') as f: f.write(buffer.getbuffer())
        self.current['archive_bytes']=buffer.tell()
    def feedback_json(self,path,value):
        with self.stage('json_serialization'): text=json.dumps(value,indent=2,allow_nan=False)+'\n'
        with self.stage('json_write'): path.write_text(text,encoding='utf-8')
    def report(self):
        gc.callbacks.remove(self.gc_callback)
        return dict(schema_version=1,scope='All stages remain inside full rollout cost; initial step0 excluded from rollout. Synchronous evidence writes included. Host wall/CPU intervals, not isolated GPU kernel durations.',events=self.events,gc_events=self.gc_events,gc_threshold=self.gc_threshold)
