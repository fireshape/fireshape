import fireshape as fs
from firedrake import split, FacetNormal, inner, ds, Constant, dx


class EnergyRecovery(fs.Objective):

    def __init__(self, pde_solver, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.pde_solver = pde_solver

    def value_form(self):
        (u, p) = split(self.pde_solver.solution)
        n = FacetNormal(self.Q.mesh_m)
        return -inner(u, n) * (p + 0.5 * inner(u, u)**2) * ds

    def derivative_form(self, deformation):
        # only right when only the noslip boundary is deformed
        return inner(Constant(0), deformation[0]) * dx
