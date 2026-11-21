"""Observed-load admission and continuous audit; never changes foreign processes.

This is not exclusive-device reservation: sub-sample bursts remain unobservable.
All thresholds are frozen before launch and raw observations are retained.
"""
import json,os,subprocess,threading,time

def query(args):
    return subprocess.check_output(['nvidia-smi']+args,text=True,timeout=10).strip()

def identity(pid):
    try:
        with open('/proc/'+str(pid)+'/stat') as stream:raw=stream.read()
        return raw[raw.rfind(')')+2:].split()[19]
    except FileNotFoundError:return None

def snapshot():
    raw=query(['--query-gpu=uuid,utilization.gpu,memory.used,temperature.gpu,clocks.current.sm,power.draw','--format=csv,noheader,nounits'])
    if len(raw.splitlines())!=1:raise RuntimeError('Controlled protocol currently requires one GPU')
    fields=[s.strip() for s in raw.split(',')]
    gpu=dict(zip(('uuid','utilization','memory_mib','temperature_c','clock_mhz','power_w'),fields))
    apps=[]
    for line in query(['--query-compute-apps=pid,process_name,used_memory','--format=csv,noheader,nounits']).splitlines():
        pid,name,memory=[s.strip() for s in line.split(',')]
        apps.append(dict(pid=int(pid),name=name,memory_mib=float(memory),start_ticks=identity(pid)))
    return dict(monotonic_s=time.monotonic(),gpu=gpu,apps=apps)

def violations(sample,baseline,owned_pids,activity):
    failures=[]
    for app in sample['apps']:
        if app['pid'] in owned_pids:continue
        before=baseline.get(app['pid'])
        if before is None or before['start_ticks']!=app['start_ticks']:
            failures.append('unregistered compute process '+str(app['pid']))
        elif abs(app['memory_mib']-before['memory_mib'])>16:
            failures.append('background memory changed '+str(app['pid']))
    for pid,sm in activity:
        if pid not in owned_pids and sm>0:failures.append('background SM activity '+str(pid))
    return failures

class LoadGuard:
    def __init__(self,output):
        self.output=output;self.samples=[];self.errors=[];self.stop_event=threading.Event();self.group=None
        self.policy=dict(idle_utilization_max_percent=5,admission_samples=10,admission_timeout_s=120,
            background_memory_tolerance_mib=16,background_sm_max_percent=0,
            scope='Observed low background load; not exclusive GPU reservation. pmon dash means no numeric sample, not proof of zero utilization.')

    def admit(self):
        self.admission=[];accepted=False;deadline=time.monotonic()+120
        while time.monotonic()<deadline:
            self.admission.append(snapshot())
            window=self.admission[-10:]
            self.baseline={a['pid']:a for a in window[0]['apps']}
            if len(window)==10 and all(float(s['gpu']['utilization'])<=5 and
                    not violations(s,self.baseline,set(),[]) for s in window):
                accepted=True;break
            time.sleep(1)
        if not accepted:self.errors.append('No stable 10-sample idle window within 120 seconds')
        self.admission_window_start=len(self.admission)-10 if accepted else None
        self.save()
        if self.errors:raise RuntimeError('GPU load admission failed; see gpu-load.json')

    def start(self,group):
        self.group=group;self.thread=threading.Thread(target=self.monitor,daemon=True);self.thread.start()

    def monitor(self):
        while not self.stop_event.is_set():
            try:
                sample=snapshot();owned=set()
                for app in sample['apps']:
                    try:
                        if os.getpgid(app['pid'])==self.group:owned.add(app['pid'])
                    except ProcessLookupError:pass
                raw=query(['pmon','-c','1','-s','u']);activity=[]
                for line in raw.splitlines():
                    f=line.split()
                    if len(f)>4 and f[0].isdigit() and f[1].isdigit() and f[2]=='C' and f[3].isdigit():
                        pid=int(f[1]);activity.append((pid,int(f[3])))
                        try:
                            if os.getpgid(pid)==self.group:owned.add(pid)
                        except ProcessLookupError:pass
                sample.update(pmon=raw,owned_pids=sorted(owned))
                sample['violations']=violations(sample,self.baseline,owned,activity)
                if sample['gpu']['uuid']!=self.admission[0]['gpu']['uuid']:sample['violations'].append('GPU identity changed')
                self.errors+=sample['violations'];self.samples.append(sample)
                with (self.output/'gpu-load.jsonl').open('a') as stream:stream.write(json.dumps(sample)+'\n')
            except Exception as exc:self.errors.append(repr(exc))
            self.stop_event.wait(.25)

    def finish(self):
        self.stop_event.set()
        if hasattr(self,'thread'):self.thread.join(timeout=25)
        if hasattr(self,'thread') and self.thread.is_alive():self.errors.append('monitor did not stop')
        if not self.samples:self.errors.append('no in-run observations')
        self.save()

    def save(self):
        result=dict(policy=self.policy,admission=getattr(self,'admission',[]),
                    admission_window_start=getattr(self,'admission_window_start',None),
                    observations=len(self.samples),errors=sorted(set(self.errors)),
                    observed_load_passed=not self.errors and bool(self.samples))
        (self.output/'gpu-load.json').write_text(json.dumps(result,indent=2)+'\n')
