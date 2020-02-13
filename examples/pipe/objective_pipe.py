import firedrake as fd
from fireshape import ShapeObjective
from PDEconstraint_pipe import NavierStokesSolver
import numpy as np


class PipeObjective(ShapeObjective):
    """L2 tracking functional for Poisson problem."""

    def __init__(self, pde_solver: NavierStokesSolver, *args,  **kwargs):
        super().__init__(*args, **kwargs)
        self.pde_solver = pde_solver

    def value_form(self):
        """Evaluate misfit functional."""
        if self.pde_solver.failed_to_solve: #is this a good solution
            return np.nan                   #is it a problem when we differeniate?
        else:
            nu = self.pde_solver.viscosity
            z = self.pde_solver.solution
            u, p = fd.split(z)
            return nu * fd.inner(fd.grad(u), fd.grad(u)) * fd.dx
