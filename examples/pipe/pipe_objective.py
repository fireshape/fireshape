import firedrake as fd
from fireshape import ShapeObjective
from pipe_PDEconstraint import NavierStokesSolver

class PipeObjective(ShapeObjective):
    """L2 tracking functional for Poisson problem."""
    def __init__(self, pde_solver: NavierStokesSolver, *args,  **kwargs):
        super().__init__(*args, **kwargs)
        self.pde_solver = pde_solver

    def value_form(self):
        """Evaluate misfit functional."""
        nu = self.pde_solver.viscosity
        z = self.pde_solver.solution
        u, p = fd.split(z)
        return nu * fd.inner(fd.grad(u), fd.grad(u)) * fd.dx
