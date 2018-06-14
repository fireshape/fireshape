import firedrake as fd
from ..objective import ShapeObjective
from .fluid_solvers import FluidSolver, StokesSolver


__all__ = ["EnergyObjective"]


class EnergyObjective(ShapeObjective):
    """Energy functional for fluid problems."""
    def __init__(self, pde_solver: FluidSolver, *args,  **kwargs):
        super().__init__(*args, **kwargs)
        self.pde_solver = pde_solver

    def solve_adjoint(self):
        super().solve_adjoint()
        # if isinstance(self.pde_solver, StokesSolver):
        #     """
        #     In the Stokes case we can calculate the adjoint by hand
        #     """
        #     (u, p) = self.pde_solver.solution.split()
        #     (v, q) = self.pde_solver.solution_adj.split()
        #     q.assign(p)
        #     q *= -1
        #     v *= 0
        #     pass
        # else:
        #     super().solve_adjoint()

    def value_form(self):
        """Evaluate misfit functional."""
        u = fd.split(self.pde_solver.solution)[0]
        nu = self.pde_solver.nu
        return 0.5 * nu * fd.inner(fd.grad(u), fd.grad(u)) * fd.dx

    def derivative_form(self, deformation):
        """Shape directional derivative of misfit functional wrt deformation."""
        u = self.pde_solver.solution.split()[0]
        w = deformation
        nu = self.pde_solver.nu
        deriv = 0.5 * nu * fd.inner(fd.grad(u), fd.grad(u)) * fd.div(w) * fd.dx
        deriv -= nu * fd.inner(fd.grad(u)*fd.grad(w), fd.grad(u)) * fd.dx
        return deriv
