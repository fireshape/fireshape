import firedrake as fd
from fireshape import ShapeObjective
from L2tracking_PDEconstraint import PoissonSolver


class L2trackingObjective(ShapeObjective):
    """L2 tracking functional for Poisson problem."""

    def __init__(self, pde_solver: PoissonSolver, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.u = pde_solver.solution

        # target function, exact soln is disc of radius 0.6 centered at
        # (0.5,0.5)
        (x, y) = fd.SpatialCoordinate(self.Q.mesh_m)
        self.u_target = 0.36 - (x-0.5)*(x-0.5) - (y-0.5)*(y-0.5)

    def value_form(self):
        """Evaluate misfit functional."""
        u = self.u
        return (u - self.u_target)**2 * fd.dx
