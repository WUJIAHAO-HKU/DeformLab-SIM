"""Experimental graded Radau contact window; empty sets retain exact free damping.

The grid is fixed as t=dt*(i/N)^3 and refined with the unchanged engineering policy.
This is a diagnostic Cartesian native window, not a general production solver.
"""
import json
import numpy as np
from .native_contact_aware_damping_probe_v2 import NativeCoupledDampingProbe as ContactAware


class NativeCoupledDampingProbe(ContactAware):
    def choose(self,mass,damping,velocity,constraint,bound,dt,substeps):
        from contact_damping_spectral_radau import SpectralRadau
        input_path=self.dest/'preselection-problem-v1.npz'
        if input_path.exists():raise ValueError('Fresh preselection evidence required')
        # Persist the actual queried full problem BEFORE any potentially failing solve.
        np.savez_compressed(input_path,mass=mass,damping=damping,velocity=velocity,
                            constraint=constraint,bound=bound,dt_s=dt)
        if not len(constraint):return super().choose(mass,damping,velocity,constraint,bound,dt,substeps)
        solver=SpectralRadau(mass,damping,velocity,constraint,bound,dt)
        path=self.dest/'adaptive-selection-v1.json'
        if path.exists():raise ValueError('Fresh selection evidence required')
        def observe(record):
            record.update(step=self.step,native_write_has_occurred=False,engineering_tolerance_only=True,
                          grid_power=3.,method='spectral-Radau-IIA-2-stage-order-3')
            path.write_text(json.dumps(record,indent=2,allow_nan=False)+'\n')
        def refine(m,D,v,C,b,t,n):
            r=solver.integrate(n,3.)
            if r['physical_frame_momentum_residual']>1e-10 or r['spectral_decomposition_residual']>1e-10:
                raise RuntimeError('Modal reconstruction fails original-coordinate equation')
            return r
        result=self.select(mass,damping,velocity,constraint,bound,dt,self.policy,observe,solver=refine)
        result['adaptive_selection']['grid_power']=3.
        self.substeps=result['substeps']
        return result
