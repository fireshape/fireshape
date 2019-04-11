import firedrake as fd
from fireshape import PdeConstraint

class PoissonSolver(PdeConstraint):
    """A Poisson BVP with hom DirBC as PDE constraint."""
    def __init__(self, mesh_m):
        super().__init__()
        self.mesh_m = mesh_m

        # Setup problem
        self.V = fd.FunctionSpace(self.mesh_m, "CG", 1)

        # Preallocate solution variables for state and adjoint equations
        self.solution = fd.Function(self.V, name="State")
        self.testfunction = fd.TestFunction(self.V)
        self.solution_adj = fd.Function(self.V, name="Adjoint")

        # Weak form of Poisson problem
        u = self.solution
        v = self.testfunction
        self.f = fd.Constant(4.)
        self.F = (fd.inner(fd.grad(u), fd.grad(v)) - self.f * v) * fd.dx
        self.bcs = fd.DirichletBC(self.V, 0., "on_boundary")

        # PDE-solver parameters
        self.nsp = None
        self.params = {
                "ksp_type": "cg",
                "mat_type": "aij",
                "pc_type": "hypre",
                "pc_factor_mat_solver_package": "boomerang",
                "ksp_rtol": 1e-11,
                "ksp_atol": 1e-11,
                "ksp_stol": 1e-15,
                      }

        stateproblem = fd.NonlinearVariationalProblem(self.F, self.solution, bcs=self.bcs)
        self.statesolver = fd.NonlinearVariationalSolver(stateproblem, solver_parameters=self.params)
