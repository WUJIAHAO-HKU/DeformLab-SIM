"""Native midpoint material composition; no material-law or gamma change.

The midpoint candidate publishes on odd native half steps; the matched endpoint
control publishes on even steps. Each material interval is twice the native
interval. Other steps and the verification tail perform no material write.
Existing publication, work and next-entry guards are retained.
"""
import json
from .native_objective_damping_probe_v6 import NativeCoupledDampingProbe


class CenteredProbe(NativeCoupledDampingProbe):
    def choose(self, mass, damping, velocity, constraint, bound, dt, substeps):
        result = super().choose(mass, damping, velocity, constraint, bound, 2*dt, substeps)
        self.composition_dt = 2*dt
        return result


class CenteredDampingWindow:
    def __init__(self, root, base, port, runtime, destination, task_frames, substeps=64, placement='midpoint'):
        from centered_damping_schedule import CenteredSchedule
        self.root = root; self.base = base; self.port = port; self.runtime = runtime
        self.dest = destination; self.substeps = substeps
        self.schedule = CenteredSchedule(runtime.config['solver']['hz'], task_frames, placement)
        self.probes = []; self.native_only_steps = []

    def before_native(self, step):
        self.schedule.before(step)
        if self.probes:
            self.probes[-1].before_native(step)

    def apply(self):
        step = self.runtime.steps+1
        active = self.schedule.after(step)
        if not active:
            # Do not call the legacy base: it would apply extra physical damping.
            self.native_only_steps.append(step)
            return
        dest = self.dest/'coupled-damping-window'/f'frame-{step}'
        dest.mkdir(parents=True, exist_ok=False)
        probe = CenteredProbe(self.root, self.base, self.port, self.runtime, dest, step, self.substeps)
        self.probes.append(probe)
        probe.apply()
        if probe.record['native_velocity_publications'] != len(self.probes):
            raise RuntimeError('Centered native publication count mismatch')
        if abs(probe.composition_dt-self.schedule.damping_dt) > 1e-15:
            raise RuntimeError('Actual damping duration differs from composition')
        probe.record.update(composition=self.schedule.method, placement=self.schedule.placement,
                            native_dt_s=1/self.schedule.native_hz,
                            damping_dt_s=probe.composition_dt,
                            publication_native_time_s=step/self.schedule.native_hz,
                            macro_interval_s=self.schedule.macro_interval(step),
                            second_order_verified=False)
        probe.publication_path.write_text(json.dumps(probe.record, indent=2, allow_nan=False)+'\n')

    def report(self):
        records = [p.record for p in self.probes]
        continued = all(r and r.get('passed') is True and 'next_frame_before_native' in r
                        and all(v for k, v in r['next_frame_before_native'].items() if k != 'step')
                        for r in records)
        expected_native_only = self.schedule.native_only_steps
        accepted = (self.schedule.complete() and self.runtime.steps == self.schedule.task_frames+1
                    and self.base.steps == 0 and continued
                    and len(records) == len(self.schedule.publication_steps)
                    and self.native_only_steps == expected_native_only)
        return dict(accepted=bool(accepted), scope=__doc__, steps=self.runtime.steps,
                    integrator=self.schedule.method, placement=self.schedule.placement,
                    requested_gamma=self.base.gamma, native_gamma=0.,
                    native_hz=self.schedule.native_hz, macro_hz=self.schedule.native_hz/2,
                    damping_dt_s=self.schedule.damping_dt,
                    total_material_time_s=len(records)*self.schedule.damping_dt,
                    task_time_s=self.schedule.task_frames/self.schedule.native_hz,
                    intervention_steps=self.schedule.publication_steps,
                    native_only_steps=self.native_only_steps, publications=records,
                    legacy_damping_calls=self.base.steps, platform_acceptance=False,
                    second_order_verified=False, equal_budget=False,
                    actuator_work_closed=False, all_native_entry_caches_audited=False)
