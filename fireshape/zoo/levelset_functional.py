import firedrake as fd
from ..objective import ShapeObjective


__all__ = ["LevelsetFunctional"]


class LevelsetFunctional(ShapeObjective):
    """
    Implementation of level-set shape functional.

    Optima are zero-levels of ufl function f.
    """
    def __init__(self, f, *args, measure=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.f = f
        if measure is None:
            self.measure = fd.dx(domain=self.Q.mesh_m, degree=30)
        else:
            self.measure = measure

    def value_form(self):
        return self.f * self.measure

    def derivative_form(self, v):
        X = fd.SpatialCoordinate(self.Q.mesh_m)
        return fd.derivative(self.value_form(), X, v)
