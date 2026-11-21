"""Explicit quadrature time for a target held during one outer physics interval."""
import math

POLICIES=('endpoint','midpoint')

def sample_time(completed_steps,hz,policy='endpoint'):
    if type(completed_steps) is not int or completed_steps<0:
        raise ValueError('Completed step count must be a nonnegative integer')
    if isinstance(hz,bool) or not isinstance(hz,(int,float)) or not math.isfinite(hz) or hz<=0:
        raise ValueError('Positive finite frequency required')
    if policy not in POLICIES:
        raise ValueError('Unknown held-target quadrature policy')
    return (completed_steps+(1. if policy=='endpoint' else .5))/hz
