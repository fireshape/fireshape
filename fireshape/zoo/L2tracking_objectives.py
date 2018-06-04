import firedrake as fd
from ..objective import Objective
from .L2tracking_solvers import PoissonSolver
__all__ = ["L2trackingObjective"]


class L2trackingOjective(Objective):
    """L2 tracking functional for Poisson problem."""
    def __init__(self, pde_solver: PoissonSolver, *args,  **kwargs):
        super().__init__(*args, **kwargs)
        self.pde_solver = pde_solver
        (x, y) = fd.SpatialCoordinate(pde_solver.mesh_m)
        self.u_target = (x-0.5)*(x-0.5) + (y-0.5)*(y-0.5) - 1

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
        return deriv
