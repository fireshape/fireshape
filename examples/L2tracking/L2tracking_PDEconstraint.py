import firedrake as fd
import firedrake_adjoint as fda
from fireshape import PdeConstraint


class PoissonSolver(PdeConstraint):
    """A Poisson BVP with hom DirBC as PDE constraint."""
    def __init__(self, mesh_m):
        super().__init__()
        self.mesh_m = mesh_m

        # Setup problem
        self.V = fd.FunctionSpace(self.mesh_m, "CG", 1)

        # Preallocate solution variables for state and adjoint equations
        self.solution = fda.Function(self.V, name="State")

        # Weak form of Poisson problem
        u = self.solution
        v = fd.TestFunction(self.V)
        self.f = fda.Constant(4.)
        self.F = (fd.inner(fd.grad(u), fd.grad(v)) - self.f * v) * fd.dx
        self.bcs = fda.DirichletBC(self.V, 0., "on_boundary")

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

        stateproblem = fda.NonlinearVariationalProblem(
            self.F, self.solution, bcs=self.bcs)
        self.solver = fda.NonlinearVariationalSolver(
            stateproblem, solver_parameters=self.params)

    def solve(self):
        super().solve()
        self.solver.solve()
