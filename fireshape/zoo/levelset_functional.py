import firedrake as fd
from ..objective import ShapeObjective

__all__ = ["LevelsetFunctional", "VolumeFunctional"]


class LevelsetFunctional(ShapeObjective):
    """
    Implementation of level-set shape functional.

    Optima are zero-levels of ufl function f.
    """
    def __init__(self, Q, f, usecb=False, *args, **kwargs):
        super().__init__(Q, *args, **kwargs)
        self.f = f
        Vdet = fd.FunctionSpace(self.Q.mesh_r, "DG", 0)
        self.detDT = fd.Function(Vdet)
        if usecb:  # if True, store meshes in soln.pvd
            out = fd.VTKFile("soln.pvd")
            self.cb = lambda: out.write(Q.mesh_m.coordinates)

    def value_form(self):
        self.detDT.interpolate(fd.det(fd.grad(self.Q.T)))
        mesh_is_fine = min(self.detDT.vector()) > 0.01
        if mesh_is_fine:
            return self.f * fd.dx(domain=self.Q.mesh_m)
        else:
            from pyadjoint.adjfloat import AdjFloat
            import numpy as np
            return AdjFloat(np.NAN) * fd.dx(domain=self.Q.mesh_m)


class VolumeFunctional(LevelsetFunctional):
    """
    Implementation of volume shape functional.

    Returns the volume of the domain.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(fd.Constant(1.0), *args, **kwargs)
