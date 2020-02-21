import firedrake as fd
from fireshape import PdeConstraint


class PoissonSolver(PdeConstraint):
    """A Poisson BVP with hom DirBC as PDE constraint."""
    def __init__(self, mesh_m):
        super().__init__()

        # Setup problem
        V = fd.FunctionSpace(mesh_m, "CG", 1)

        # Weak form of Poisson problem
        u = fd.Function(V, name="State")
        v = fd.TestFunction(V)
        f = fd.Constant(4.)
        self.F = (fd.inner(fd.grad(u), fd.grad(v)) - f * v) * fd.dx
        self.bcs = fd.DirichletBC(V, 0., "on_boundary")

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

        # problem = fd.NonlinearVariationalProblem(
        #     self.F, self.solution, bcs=self.bcs)
        # self.solver = fd.NonlinearVariationalSolver(
        #     problem, solver_parameters=self.params)

    def solve(self):
        super().solve()
        fd.solve(self.F == 0, self.solution, bcs=self.bcs,
                 solver_parameters=self.params)
        # self.solver.solve()
