"""Publish contact-aware adaptive damping on a declared consecutive native-frame window."""
from .native_contact_aware_damping_probe import NativeCoupledDampingProbe


class CoupledDampingWindow:
    def __init__(self,root,base,port,runtime,destination,first=128,last=129,substeps=64):
        if not 1<=first<=last:raise ValueError('Ordered positive frame window required')
        self.root=root;self.base=base;self.port=port;self.runtime=runtime;self.dest=destination
        self.first=first;self.last=last;self.substeps=substeps;self.probes=[]

    def before_native(self,step):
        if self.probes:self.probes[-1].before_native(step)

    def apply(self):
        step=self.runtime.steps+1
        if not self.first<=step<=self.last:
            self.base.apply();self.base.ledger[-1]['step']=step;return
        if step!=self.first+len(self.probes):raise RuntimeError('Repeated or missing coupled frame')
        dest=self.dest/'coupled-damping-window'/f'frame-{step}'
        dest.mkdir(parents=True,exist_ok=False)
        probe=NativeCoupledDampingProbe(self.root,self.base,self.port,self.runtime,dest,step,self.substeps)
        self.probes.append(probe);probe.apply()
        if probe.record['native_velocity_publications']!=len(self.probes):raise RuntimeError('Native publication count mismatch')

    def report(self):
        other=self.base.report();count=self.last-self.first+1
        other['accepted']=self.base.steps==self.runtime.steps-count
        records=[p.record for p in self.probes]
        passed=len(records)==count and all(r and r['passed'] and all(v for k,v in r.get('next_frame_before_native',{}).items() if k!='step') and 'next_frame_before_native' in r for r in records)
        return dict(accepted=bool(passed and other['accepted']),steps=self.runtime.steps,scope=__doc__,
            integrator='consecutive-contact-aware-damping-window',intervention_steps=list(range(self.first,self.last+1)),
            initial_reference_substeps=self.substeps,selected_substeps={p.step:p.record['substeps'] for p in self.probes if p.record},requested_gamma=self.base.gamma,exponential_other_frames=other,
            publications=records,platform_acceptance=False,whole_horizon_coupled_damping=False)
