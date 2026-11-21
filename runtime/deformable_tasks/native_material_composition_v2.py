"""Material composition retaining terminal contact correction at every half step.

Material-active transactions use the requested gamma for two native intervals.
Other task transactions use the already validated zero-D contact solve, with
explicit impulse/work accounting. The original base is never mutated. The one
verification tail retains the pre-existing out-of-task base update contract.
"""
import json
from .native_objective_damping_probe_v6 import NativeCoupledDampingProbe
from .native_objective_damping_window_v6 import CoupledDampingWindow


def material_active(step, placement):
    if type(step) is not int or step < 1 or placement not in ('midpoint','endpoint'):
        raise ValueError('Positive native step and declared material placement required')
    return step % 2 == (1 if placement == 'midpoint' else 0)


class ContactOnlyBase:
    """Read-through native buffers with exactly zero material coefficient."""
    gamma = 0.
    def __init__(self, base): self._original_base = base
    def __getattr__(self, name): return getattr(self._original_base, name)
    def apply(self): raise RuntimeError('Contact-only branch must use the coupled transaction, never legacy damping')


class MaterialCompositionProbe(NativeCoupledDampingProbe):
    def __init__(self, root, base, port, runtime, destination, step, substeps, placement):
        self.material_active = material_active(step, placement)
        self.requested_gamma = base.gamma
        self.placement = placement
        selected = base if self.material_active else ContactOnlyBase(base)
        super().__init__(root, selected, port, runtime, destination, step, substeps)

    def choose(self, mass, damping, velocity, constraint, bound, dt, substeps):
        self.actual_solver_interval = dt*(2 if self.material_active else 1)
        return super().choose(mass, damping, velocity, constraint, bound, self.actual_solver_interval, substeps)

    def apply(self):
        super().apply()
        gamma = self.requested_gamma if self.material_active else 0.
        if self.record['material_gamma'] != gamma:
            raise RuntimeError('Actual operator coefficient differs from declared composition role')
        if not self.material_active and self.record['physical_damping_work_J'] != 0.:
            raise RuntimeError('Contact-only correction applied material damping')
        native_hz = self.runtime.config['solver']['hz']
        self.record.update(material_composition='contact-every-half-step-v2',placement=self.placement,
            material_active=self.material_active,requested_material_gamma=self.requested_gamma,
            native_dt_s=1/native_hz,solver_interval_s=self.actual_solver_interval,
            physical_material_interval_s=2/native_hz if self.material_active else 0.,
            contact_only_projection=not self.material_active,
            extra_contact_integration_loss_J=self.record['integration_dissipation_J'] if not self.material_active else None,
            second_order_verified=False,platform_acceptance=False)
        self.publication_path.write_text(json.dumps(self.record,indent=2,allow_nan=False)+'\n')


class MaterialCompositionWindow(CoupledDampingWindow):
    def __init__(self,root,base,port,runtime,destination,first=1,last=1152,substeps=64,placement='midpoint'):
        if first != 1 or type(last) is not int or last < 2 or last % 2:
            raise ValueError('Complete pairs of native half steps required')
        material_active(1,placement)
        self.placement=placement
        super().__init__(root,base,port,runtime,destination,first,last,substeps)

    def apply(self):
        step=self.runtime.steps+1
        if step == self.last+1:
            self.base.apply();self.base.ledger[-1]['step']=step;return
        if not self.first <= step <= self.last or step != self.first+len(self.probes):
            raise RuntimeError('Every task half step requires exactly one coupled transaction')
        dest=self.dest/'coupled-damping-window'/f'frame-{step}'
        dest.mkdir(parents=True,exist_ok=False)
        probe=MaterialCompositionProbe(self.root,self.base,self.port,self.runtime,dest,step,self.substeps,self.placement)
        self.probes.append(probe);probe.apply()
        if probe.record['native_velocity_publications'] != len(self.probes):
            raise RuntimeError('Material/contact-only native publication count mismatch')

    def report(self):
        result=super().report()
        material_steps=[p.step for p in self.probes if p.material_active]
        contact_steps=[p.step for p in self.probes if not p.material_active]
        expected_material=[s for s in range(1,self.last+1) if material_active(s,self.placement)]
        material_time=sum(p.record['physical_material_interval_s'] for p in self.probes if p.record)
        expected_time=self.last/self.runtime.config['solver']['hz']
        valid=(material_steps==expected_material and len(contact_steps)==self.last//2
            and abs(material_time-expected_time)<1e-12)
        result.update(accepted=bool(result['accepted'] and valid),scope=__doc__,
            integrator='native-material-composition-contact-every-half-step-v2',placement=self.placement,
            material_active_steps=material_steps,contact_only_steps=contact_steps,
            physical_material_time_s=material_time,task_time_s=expected_time,
            verification_tail_role='original-base-update-outside-physical-task',
            every_task_half_step_coupled_contact=True,second_order_verified=False,equal_budget=False,
            platform_acceptance=False)
        return result
