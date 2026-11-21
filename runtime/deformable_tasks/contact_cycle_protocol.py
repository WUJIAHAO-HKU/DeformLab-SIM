"""Explicit short diagnostic cycle, separate from the original slow convergence protocol."""
import math


def smooth(x):
    x=max(0.,min(1.,float(x)));return x*x*x*(10+x*(-15+6*x))


def compression_cycle(t):
    if not math.isfinite(t):raise ValueError('Finite simulation time required')
    return smooth(t/.16)*(1-smooth((t-.20)/.16))+smooth((t-.40)/.16)
