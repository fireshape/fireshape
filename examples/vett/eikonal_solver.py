import fireshape as fs
from firedrake import FunctionSpace, Function, TestFunction, DirichletBC, \
    Constant, grad, inner, dx, NonlinearVariationalProblem, \
    NonlinearVariationalSolver, ConvergenceError, div, nabla_grad


class EikonalSolver(fs.PdeConstraint):

    def __init__(self, m_mesh, bids, degree=1):
        super().__init__()
        self.nsp = None
        V = FunctionSpace(m_mesh, "CG", degree)
        self.V = V
        d = Function(V, name="eikonal_state")
        self.solution = d
        self.solution_adj = Function(V, name="eikonal_adjoint")
        v = TestFunction(V)
        self.bcs = [DirichletBC(V, 0.0, bids)]
        self.eps = Constant(1.0)
        self.F = inner(grad(d), grad(d)) * v * dx \
            + self.eps * inner(grad(d), grad(v)) * dx \
            - Constant(1.0) * v * dx
        problem = NonlinearVariationalProblem(self.F, d, bcs=self.bcs)
        self.params = {
            "ksp_type": "preonly",
            "pc_type": "lu",
            "pc_factor_mat_solver_type": "mumps",
            "ksp_max_it": 2,
            "snes_max_it": 10,
            "snes_tol": 1e-10
        }
        solver = NonlinearVariationalSolver(problem,
                                            solver_parameters=self.params)
        self.solver = solver
        self.first_solve = True

    def solve(self):
        super().solve()
        d = self.solution
        if self.first_solve:
            for i in range(3):
                d_last = d.copy(deepcopy=True)
                self.eps.assign(10**(-i))
                try:
                    self.solver.solve()
                    print("Eikonal solved for eps=%f" % self.eps.values()[0])
                except ConvergenceError:
                    print("Can't solve the damn eikonal equation")
                    self.eps.assign(10**(-i+1))
                    d.assign(d_last)
                    break
            self.first_solve = False
        else:
            self.solver.solve()
        return d

    def derivative_form(self, deformation):
        w = deformation
        X = SpatialCoordinate(w.ufl_domain())
        F = self.F
        F = replace(self.F, {self.F.arguments()[0]: self.solution_adj})
        return derivative(F, X, w)
