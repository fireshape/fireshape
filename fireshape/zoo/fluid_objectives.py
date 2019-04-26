import firedrake as fd
from ..objective import ShapeObjective
from .fluid_solvers import FluidSolver


__all__ = ["EnergyObjective"]


class EnergyObjective(ShapeObjective):
    """Energy functional for fluid problems."""
    def __init__(self, pde_solver: FluidSolver, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.pde_solver = pde_solver

    def value_form(self):
        """Evaluate misfit functional."""
        u = fd.split(self.pde_solver.solution)[0]
        nu = self.pde_solver.nu
        return 0.5 * nu * fd.inner(fd.grad(u), fd.grad(u)) * fd.dx
