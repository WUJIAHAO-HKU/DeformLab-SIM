"""Select a Radau budget on the actual frame, then use the verified native velocity transaction."""
import json
from .native_coupled_damping_radau_probe import NativeCoupledDampingProbe as RadauProbe


class NativeCoupledDampingProbe(RadauProbe):
    def __init__(self,root,base,port,runtime,destination,step=128,substeps=64):
        super().__init__(root,base,port,runtime,destination,step,substeps)
        from adaptive_radau_damping import select,DEFAULT_POLICY
        self.select=select;self.policy=dict(DEFAULT_POLICY,initial_substeps=substeps)
        self.refine=self.adaptive_refine

    def adaptive_refine(self,mass,damping,velocity,constraint,bound,dt,substeps):
        path=self.dest/'adaptive-selection-v1.json'
        if path.exists():raise ValueError('Fresh per-frame selection evidence required')
        def observe(record):
            record.update(step=self.step,native_write_has_occurred=False,engineering_tolerance_only=True)
            path.write_text(json.dumps(record,indent=2,allow_nan=False)+'\n')
        result=self.select(mass,damping,velocity,constraint,bound,dt,self.policy,observe)
        self.substeps=result['substeps'];self.selection=result['adaptive_selection']
        return result

    def apply(self):
        super().apply()
        if self.record is not None:
            self.record['adaptive_selection']=self.selection
            self.publication_path.write_text(json.dumps(self.record,indent=2,allow_nan=False)+'\n')
