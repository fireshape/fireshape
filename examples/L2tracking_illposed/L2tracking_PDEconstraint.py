from math import log
import firedrake as fd
from fireshape import PdeConstraint


class PoissonSolver(PdeConstraint):
    """A Poisson BVP with DirBC as PDE constraint."""
    def __init__(self, mesh_m, rt, ct):
        super().__init__()
        self.mesh_m = mesh_m

        # Setup problem
        V = fd.FunctionSpace(self.mesh_m, "CG", 1)

        # Weak form of Poisson problem
        u = fd.Function(V, name="State")
        v = fd.TestFunction(V)
        self.F = fd.inner(fd.grad(u), fd.grad(v)) * fd.dx

        # Dirichlet boundary conditions
        X = fd.SpatialCoordinate(self.mesh_m)
        shift = fd.as_vector([X[0] - ct[0], X[1] - ct[1]])
        self.u_target = -fd.ln(fd.sqrt(fd.inner(shift, shift))) / log(rt) + 1.
        self.bcs = [fd.DirichletBC(V, 0., 1),
                    fd.DirichletBC(V, self.u_target, 2)]

        # PDE-solver parameters
        self.params = {
            "ksp_type": "cg",
            "mat_type": "aij",
            "pc_type": "hypre",
            "pc_factor_mat_solver_package": "boomerang",
            "ksp_rtol": 1e-11,
            "ksp_atol": 1e-11,
            "ksp_stol": 1e-15,
        }

        self.solution = u

    def solve(self):
        super().solve()
        fd.solve(self.F == 0, self.solution, bcs=self.bcs,
                 solver_parameters=self.params)
