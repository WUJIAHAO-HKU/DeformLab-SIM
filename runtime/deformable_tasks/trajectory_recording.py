"""Explicit full-rate cloth recording; diagnostic timing is never accepted."""
import os
from .cloth_split_backend import BACKENDS

def requested(task,backend,rendered=False,suite=False,timing_only=False):
 from .cloth_geometry_checkpoint import requested as geometry_requested
 geometry_requested(task,backend)
 value=os.environ.get('DEFORMABLE_CLOTH_FULL_RATE','0')
 if value not in ('0','1'):raise ValueError('Full-rate recording must be explicit 0 or 1')
 if value=='0':return False
 if task not in ('cloth_grip','cloth_fold') or backend not in ('native',*BACKENDS) or rendered or suite or timing_only:raise ValueError('Full-rate recording requires one unrendered cloth diagnostic task; no timing-only or fallback')
 return True

def stride(hz,full_rate):
 if type(hz) is not int or hz<=0:raise ValueError('Positive integer physics frequency required')
 return 1 if full_rate else max(1,hz//15)

def evidence(full_rate,record_stride,indices,steps,requested_steps):
 complete=bool(full_rate and record_stride==1 and steps==requested_steps and indices==list(range(requested_steps+1)))
 return dict(full_rate_requested=bool(full_rate),mode='every_physics_step' if full_rate else 'sparse',stride_steps=record_stride,frames=len(indices),requested_steps=requested_steps,observed_steps=steps,complete_full_rate_coverage=complete,performance_eligible=False if full_rate else None,scope='End-step nodal positions only; not every internal iteration or continuous collision detection.')
