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
        if getattr(self, "is_decoupled", False):
            # if the control is on a different mesh mesh_c,
            # then compute the determinant of DT there
            Vdet = fd.FunctionSpace(self.Q.V_c.mesh(), "DG", 0)
            self.T_c = self.Q.V_c.mesh().coordinates + self.Q.fun
        else:
            # otherwise compute detDT on mesh_r
            Vdet = fd.FunctionSpace(self.Q.mesh_r, "DG", 0)
        self.detDT = fd.Function(Vdet)
        if usecb and self.cb is None:  # if True, store meshes in soln.pvd
            out = fd.VTKFile("soln.pvd")
            self.cb = lambda: out.write(Q.mesh_m.coordinates)

    def value_form(self):
        if getattr(self, "is_decoupled", False):
            self.detDT.interpolate(fd.det(fd.grad(self.T_c)))
        else:
            self.detDT.interpolate(fd.det(fd.grad(self.Q.T)))
        # approximately check that detDT is not too small
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
