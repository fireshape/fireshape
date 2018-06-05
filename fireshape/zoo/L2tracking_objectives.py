import firedrake as fd
from ..objective import Objective
from .L2tracking_solvers import FullH1Solver

__all__ = ["L2trackingObjective"]

class L2trackingObjective(Objective):
    """L2 tracking functional for Poisson problem."""
    def __init__(self, pde_solver: FullH1Solver, *args,  **kwargs):
        super().__init__(*args, **kwargs)
        self.pde_solver = pde_solver

        (x, y) = fd.SpatialCoordinate(pde_solver.mesh_m)
        self.u_target = 0.36 - (x-0.5)*(x-0.5) - (y-0.5)*(y-0.5)

    def solve_adjoint(self):
        super().solve_adjoint()

    def value_form(self):
        """Evaluate misfit functional."""
        u = self.pde_solver.solution
        return (u - self.u_target)**2 * fd.dx

    def derivative_form(self, deformation):
        """Shape directional derivative of misfit functional wrt deformation."""
        u = self.pde_solver.solution
        w = deformation
        deriv = (u - self.u_target)**2 * fd.div(w) * fd.dx
        deriv -= 2*(u - self.u_target) * fd.inner(fd.grad(self.u_target), w) * fd.dx
        return deriv
