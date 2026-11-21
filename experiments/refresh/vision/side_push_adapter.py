"""Unmodified existing B, sparse-D and dense same-history implementations."""
from pathlib import Path
import sys
import numpy as np

class Adapter:
    def __init__(self,root,runtime,arm,build,geometry):
        self.trial=None;self.surface=None;self.stopped=False;self.records=[]
        self.runtime=runtime;self.arm=arm;self.build=Path(build)
        self.evidence=dict(arm=arm,stop_attempted=False,normal_policy_enabled=False,static_support_only=True)
        if arm not in ['A','B','D','dense','frozen_first']:raise ValueError(arm)
        sys.path.insert(0,str(self.build/'source'))
        from cp_trial_bridge import ConservativeCpTrial
        from timing_surface_bridge import RobotSurfaceLive
        from omni.physx import get_physx_interface
        sys.path.insert(0,str(Path(root)/'work/native-readback'))
        import physx_deformable_probe_controls as controls
        b=runtime.bodies[0];interface=get_physx_interface();controls.configure(interface,b['path'],0,False)
        try:
            self.trial=ConservativeCpTrial(root,interface,b['path'],'pass-through'if arm=='A'else'split-velocity')
            self.evidence['cp']=self.trial.evidence
            settings=self.trial.evidence['native_settings']
            required={'material_model':0,'youngs_modulus':runtime.config['material']['young_pa'],'poissons_ratio':.3,'linear_damping':0.,'elasticity_damping':0.,'settling_damping':0.,'settling_threshold':0.,'sleep_threshold':0.}
            assert all(k in settings and np.isclose(settings[k],v,atol=1e-7,rtol=1e-6)for k,v in required.items()),settings
            if arm in ['D','dense','frozen_first']:
                g=geometry['oracle_geometry'];center=g['bench_position_m'];half=g['bench_half_extents_m']
                assert np.allclose(half[:2],[.08,.08],atol=1e-8,rtol=0)
                self.surface=RobotSurfaceLive(self.trial,self.build/'rows.ptx',b['sim_elements'],center,center[2]+half[2],runtime.config['solver']['position_iterations'],runtime.event_reference)
                self.evidence['surface']=self.surface.evidence
        except BaseException:
            self.stop();raise
    def before(self,step):
        if self.surface:self.surface.frame(step)
    def after(self,step):
        if self.surface:self.surface.after_frame(step)
        c=self.trial.counts();expected=step*(self.runtime.config['solver']['position_iterations']+1)
        if c[:3]!=([0,expected,expected]if self.arm=='A'else[expected]*3):raise RuntimeError('CP coverage '+repr(c))
        s=self.surface.snapshot()if self.surface else [0]*16
        self.records.append([step,*c,*s])
    def finish(self):
        steps=self.runtime.steps;expected=steps*(self.runtime.config['solver']['position_iterations']+1)
        assert len(self.records)==steps
        self.evidence.update(cp_counts=self.trial.counts(),expected_rounds=expected,frame_records=len(self.records))
        if self.surface:
            self.evidence['legacy_surface_finish']=self.surface.finish(steps)
            self.evidence['surface_branch_triggered']=self.surface.snapshot()[9]>0
        else:self.evidence['surface_branch_triggered']=False
        self.stop()
        assert self.evidence.get('stop_error_count')==0,self.evidence
        return self.evidence
    def stop(self):
        if self.stopped:return
        self.stopped=True
        if self.trial is not None:
            self.evidence['stop_attempted']=True
            self.evidence['stop_error_count']=self.trial.stop()
