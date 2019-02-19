import firedrake as fd
from fireshape import PdeConstraint

__all__ = ["PoissonSolver"]
class PoissonSolver(PdeConstraint):
    """A Poisson BVP with hom DirBC as PDE constraint."""
    def __init__(self, mesh_m): 
        super().__init__()
        self.mesh_m = mesh_m

        # Setup problem
        self.V = fd.FunctionSpace(self.mesh_m, "CG", 1)

        # Preallocate solution variables for state and adjoint equations
        self.solution = fd.Function(self.V, name="State")
        self.solution_adj = fd.Function(self.V, name="Adjoint")

        # Weak form of Poisson problem
        u = self.solution
        v = fd.TestFunction(self.V)
        self.testfunction = v
        self.f = fd.Constant(4.)
        self.F = (fd.inner(fd.grad(u), fd.grad(v)) - self.f * v) * fd.dx
        self.bcs = fd.DirichletBC(self.V, 0., "on_boundary")

        # PDE-solver parameters
        self.nsp = None
        self.params = {
                "ksp_type": "fgmres",
                "mat_type": "aij",
                "pc_type": "lu",
                "pc_factor_mat_solver_package": "mumps",
                "ksp_atol": 1e-15,
                      }

        stateproblem = fd.NonlinearVariationalProblem(self.F, self.solution, bcs=self.bcs)
        self.stateproblem = fd.NonlinearVariationalSolver(stateproblem, solver_parameters=self.params)

    def solve(self):
        """Solve the state equation."""
        super().solve()
        self.stateproblem.solve()
        #fd.solve(self.F == 0, self.solution, bcs=self.bcs,
        #         solver_parameters=self.params)
        return self.solution

    def derivative_form(self, deformation):
        """Shape directional derivative of self.F wrt to w."""
        w = deformation
        u = self.solution
        p = self.solution_adj
        f = self.f
        Dw = fd.nabla_grad(w)

        deriv = (fd.div(w) * (fd.inner(fd.grad(u), fd.grad(p)) - f * p)
                - fd.inner(Dw * fd.grad(u), fd.grad(p))
                - fd.inner(fd.grad(u), Dw * fd.grad(p))
                ) * fd.dx
        return deriv
