import firedrake as fd
from ..objective import ShapeObjective


__all__ = ["LevelsetFunctional"]


class LevelsetFunctional(ShapeObjective):
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
        X = fd.SpatialCoordinate(self.Q.mesh_m)
        return fd.derivative(self.value_form(), X, v)
