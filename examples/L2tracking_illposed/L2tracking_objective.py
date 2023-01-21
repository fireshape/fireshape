import firedrake as fd
from fireshape import ShapeObjective
from L2tracking_PDEconstraint import PoissonSolver


class L2trackingObjective(ShapeObjective):
    """L2 tracking functional for Poisson problem."""

    def __init__(self, pde_solver: PoissonSolver, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Fixed subdomain B
        X = fd.SpatialCoordinate(pde_solver.mesh_m)
        self.B = fd.SubDomainData(X[0] >= 0.)

        self.u = pde_solver.solution
        self.u_target = pde_solver.u_target

    def value_form(self):
        """Evaluate misfit functional."""
        return (self.u - self.u_target)**2 * fd.dx(subdomain_data=self.B)
