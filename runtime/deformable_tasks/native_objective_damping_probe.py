"""Use the objective spectral gradient in selected native graded damping transactions only."""
from .native_graded_damping_probe import NativeCoupledDampingProbe as Graded

class NativeCoupledDampingProbe(Graded):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        from objective_directional_damping import operator
        self.operator=operator
