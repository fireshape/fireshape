import firedrake as fd
from fireshape import ShapeObjective
from L2tracking_PDEconstraint import PoissonSolver


class L2trackingObjective(ShapeObjective):
    """L2 tracking functional for Poisson problem."""
    def __init__(self, pde_solver: PoissonSolver, *args,  **kwargs):
        super().__init__(*args, **kwargs)
        self.pde_solver = pde_solver

        # target function, exact soln is disc of radius 0.6 centered at
        # (0.5,0.5)
        (x, y) = fd.SpatialCoordinate(pde_solver.mesh_m)
        self.u_target = 0.36 - (x-0.5)*(x-0.5) - (y-0.5)*(y-0.5)

    def value_form(self):
        """Evaluate misfit functional."""
        u = self.pde_solver.solution
        return (u - self.u_target)**2 * fd.dx
