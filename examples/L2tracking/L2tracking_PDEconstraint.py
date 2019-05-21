import firedrake as fd
import firedrake_adjoint as fda
from fireshape import PdeConstraint


class PoissonSolver(PdeConstraint):
    """A Poisson BVP with hom DirBC as PDE constraint."""
    def __init__(self, mesh_m):
        super().__init__()

        # Setup problem
        V = fd.FunctionSpace(mesh_m, "CG", 1)

        # Weak form of Poisson problem
        u = fda.Function(V, name="State")
        v = fd.TestFunction(V)
        f = fda.Constant(4.)
        F = (fd.inner(fd.grad(u), fd.grad(v)) - f * v) * fd.dx
        bcs = fda.DirichletBC(V, 0., "on_boundary")

        # PDE-solver parameters
        params = {
            "ksp_type": "cg",
            "mat_type": "aij",
            "pc_type": "hypre",
            "pc_factor_mat_solver_package": "boomerang",
            "ksp_rtol": 1e-11,
            "ksp_atol": 1e-11,
            "ksp_stol": 1e-15,
        }

        self.solution = u
        problem = fda.NonlinearVariationalProblem(
            F, self.solution, bcs=bcs)
        self.solver = fda.NonlinearVariationalSolver(
            problem, solver_parameters=params)

    def solve(self):
        super().solve()
        self.solver.solve()
