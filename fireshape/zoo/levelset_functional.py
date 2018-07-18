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
        return fd.div(self.f*v) * fd.dx

    def second_derivative_form(self, v, w):
        X = fd.SpatialCoordinate(v.ufl_domain())
        n = fd.FacetNormal(v.ufl_domain())
        from firedrake import inner, grad, ds
        # return inner(grad(self.f), n) * inner(v, n) * inner(w,n) * ds
        return fd.derivative(fd.derivative(self.value_form(), X, v), X, w)
