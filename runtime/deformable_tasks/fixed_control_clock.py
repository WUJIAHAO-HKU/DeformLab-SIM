"""Integer-ratio controller clock, independent of physics refinement.

This schedules held commands only; it never advances or shifts physical time.
The controller must divide the physics frequency exactly so there are no
partial control intervals or time interpolation hidden in the comparison.
"""
import math
import numbers


def sample_time(completed_steps, physics_hz, control_hz, policy='midpoint'):
    if isinstance(completed_steps,bool) or not isinstance(completed_steps,numbers.Integral) or completed_steps<0:
        raise ValueError('Nonnegative integral completed physics steps required')
    for value in (physics_hz,control_hz):
        if isinstance(value,bool) or not isinstance(value,numbers.Real) or not math.isfinite(value) or value<=0 or int(value)!=value:
            raise ValueError('Positive integral clock frequencies required')
    physics_hz,control_hz=int(physics_hz),int(control_hz)
    if physics_hz%control_hz or physics_hz<control_hz:
        raise ValueError('Physics frequency must be an integer multiple of controller frequency')
    if policy not in ('endpoint','midpoint'):
        raise ValueError('Explicit endpoint or midpoint controller sampling required')
    index=int(completed_steps)//(physics_hz//control_hz)
    return (index+(1.0 if policy=='endpoint' else .5))/control_hz
