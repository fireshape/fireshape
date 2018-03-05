import firedrake as fd
from ..objective import Objective


__all__ = ["LevelsetFunctional"]


class LevelsetFunctional(Objective):
    """
    Implementation of level-set shape functional.

    Optima are zero-levels of ufl function f.
    """
    def __init__(self, f, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.f = f

    def value_form(self):
        return self.f * fd.dx(domain=self.Q.mesh_m)

    def derivative_form(self, v):
        return fd.div(self.f*v) * fd.dx
