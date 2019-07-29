import firedrake as fd


class InteriorControlConstraint(object):

    def __init__(self, V, form, direct_solve=True):
        self.V = V
        self.direct_solve = direct_solve
        self.a = form.get_form(V)
        assert form.is_symmetric()
        self.bc = fd.DirichletBC(V, 0, "on_boundary")
        self.A = fd.assemble(self.a, bcs=self.bc, mat_type="aij")
        self.ls = fd.LinearSolver(self.A, solver_parameters=self.get_params())
        self.temp1 = fd.Function(V)
        self.temp2 = fd.Function(V)

    def get_params(self):
        """PETSc parameters to solve linear system."""
        params = {
            'ksp_rtol': 1e-11,
            'ksp_atol': 1e-11,
            'ksp_stol': 1e-16,
            'ksp_type': 'cg',
        }
        if self.direct_solve:
            params["pc_type"] = "cholesky"
            params["pc_factor_mat_solver_type"] = "mumps"
        else:
            params["pc_type"] = "hypre"
            params["pc_hypre_type"] = "boomeramg"
        return params

    def solve(self, rhs, out):
        self.ls.solve(out, rhs)

    def apply_action(self, fun, out):
        fd.assemble(fd.action(self.a, fun), tensor=out)

    def forward(self, fun, out):
        self.apply_action(fun, self.temp1)
        self.temp1 *= -1.
        self.solve(self.temp1, self.temp2)
        out += self.temp2

    def adjoint(self, rhs, out):
        self.solve(rhs, self.temp1)
        self.temp1 *= -1.
        self.apply_action(self.temp1, self.temp2)
        out += self.temp2
