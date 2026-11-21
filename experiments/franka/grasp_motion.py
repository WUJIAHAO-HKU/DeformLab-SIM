"""Dimensionless lift profiles; timing, travel, and hold windows stay unchanged."""
import math


def lift_fraction(phase, profile='quintic'):
    if not math.isfinite(phase):
        raise ValueError('Nonfinite lift phase')
    if profile not in ('linear', 'quintic'):
        raise ValueError('Unknown lift profile')
    u = min(1., max(0., float(phase)))
    if profile == 'linear':
        return u
    return u*u*u*(10.+u*(-15.+6.*u))
