import fireshape as fs
from firedrake import FunctionSpace, Function, TestFunction, DirichletBC, \
    Constant, grad, inner, dx, NonlinearVariationalProblem, \
    NonlinearVariationalSolver, ConvergenceError, div, nabla_grad


class EikonalSolver(fs.PdeConstraint):

    def __init__(self, m_mesh, bids):
        super().__init__()
        self.nsp = None
        V = FunctionSpace(m_mesh, "CG", 1)
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
            "snes_max_it": 10
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
                    print(f"Eikonal solved for eps={self.eps.values()[0]}")
                except ConvergenceError:
                    print("Can't solve the damn eikonal equation")
                    self.eps.assign(10**(-i+1))
                    d.assign(d_last)
                    break
            self.first_solve = False
        else:
            self.solver.solve()
        return d

    def shape_derivative_form(self, deformation):
        w = deformation
        d = self.solution
        v = self.solution_adj
        deriv = -2 * inner(grad(d), nabla_grad(w)*grad(d)) * v * dx
        deriv -= self.eps * inner(nabla_grad(w)*grad(d), grad(v)) * dx
        deriv -= self.eps * inner(grad(d), nabla_grad(w)*grad(v)) * dx

        deriv += inner(grad(d), grad(d)) * v * div(w) * dx
        deriv += self.eps * inner(grad(d), grad(v)) * div(w) * dx
        deriv -= v * div(w) * dx
        return deriv
