import firedrake as fd
from ..objective import ShapeObjective

__all__ = ["LevelsetFunctional", "VolumeFunctional", "SurfaceAreaFunctional"]


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


class VolumeFunctional(LevelsetFunctional):
    """
    Implementation of volume shape functional.

    Returns the volume of the domain.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(fd.Constant(1.0), *args, **kwargs)


class SurfaceAreaFunctional(ShapeObjective):
    """
    Implementation of surface area shape function.

    Optima are zero-levels of ufl function f.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def value_form(self):
        return fd.Constant(1.0) * fd.ds(domain=self.Q.mesh_m)
