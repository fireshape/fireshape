import firedrake as fd
from .bilinearform import GoodHelmholtzForm


class DirichletExtension(object):

    def __init__(self, V, form=None, fixed_dims=[], direct_solve=False):
        if isinstance(fixed_dims, int):
            fixed_dims = [fixed_dims]
        if form is None:
            form = GoodHelmholtzForm()
        self.V = V
        self.fixed_dims = fixed_dims
        self.direct_solve = direct_solve
        self.zero = fd.Constant(V.mesh().topological_dimension() * (0,))
        self.zero_fun = fd.Function(V)
        self.a = form.get_form(V)
        self.bc_fun = fd.Function(V)
        if len(self.fixed_dims) == 0:
            bcs = [fd.DirichletBC(self.V, self.bc_fun, "on_boundary")]
        else:
            bcs = []
            for i in range(self.V.mesh().topological_dimension()):
                if i in self.fixed_dims:
                    bcs.append(fd.DirichletBC(self.V.sub(i), 0, "on_boundary"))
                else:
                    bcs.append(fd.DirichletBC(self.V.sub(
                        i), self.bc_fun.sub(i), "on_boundary"))
        self.A_ext = fd.assemble(self.a, bcs=bcs, mat_type="aij")
        self.ls_ext = fd.LinearSolver(
            self.A_ext, solver_parameters=self.get_params())
        self.A_adj = fd.assemble(self.a, bcs=fd.DirichletBC(
            self.V, self.zero, "on_boundary"), mat_type="aij")
        self.ls_adj = fd.LinearSolver(self.A_adj,
                                      solver_parameters=self.get_params())

    def extend(self, bc_val, out):
        self.bc_fun.assign(bc_val)
        self.ls_ext.solve(out, self.zero_fun)

    def solve_homogeneous_adjoint(self, rhs, out):
        for i in self.fixed_dims:
            temp = rhs.sub(i)
            temp *= 0
        self.ls_adj.solve(out, rhs)

    def apply_adjoint_action(self, x, out):
        # fd.assemble(fd.action(self.a, x), tensor=out)
        out.assign(fd.assemble(fd.action(self.a, x)))

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


class NeumannExtension(object):

    def __init__(self, V, S, form=None, fixed_dims=[],
                 direct_solve=True, fixed_bids=[]):
        """
            V: VectorFunctionSpace for the deformations
            S: FunctionSpace for the control (on the boundary)
        """
        if isinstance(fixed_dims, int):
            fixed_dims = [fixed_dims]
        if form is None:
            form = GoodHelmholtzForm()
        self.V = V
        self.S = S
        self.rhs = fd.Function(V)
        self.fixed_dims = fixed_dims
        self.direct_solve = direct_solve

        bcs = []
        for dim in fixed_dims:
            bcs.append(fd.DirichletBC(self.V.sub(dim), 0, "on_boundary"))
        for bid in fixed_bids:
            bcs.append(fd.DirichletBC(self.V, 0, bid))

        self.a = form.get_form(V)
        assert form.is_symmetric()
        self.A_ext = fd.assemble(self.a, bcs=bcs, mat_type="aij")

        nsp = None
        if len(fixed_bids) == 0:
            nsp_functions = form.get_nullspace(V)
            if nsp_functions is not None:
                nsp = fd.VectorSpaceBasis(nsp_functions)
                nsp.orthonormalize()
        self.ls_ext = fd.LinearSolver(
            self.A_ext, solver_parameters=self.get_params(), nullspace=nsp,
            transpose_nullspace=nsp)

    def forward(self, ctrl, out):
        v = fd.TestFunction(self.V)
        n = fd.FacetNormal(self.V.mesh())
        fd.assemble(fd.inner(v, n) * ctrl.fun * fd.ds, tensor=self.rhs)
        self.ls_ext.solve(out, self.rhs)

    def adjoint(self, deriv, out):
        v = fd.TestFunction(self.S)
        n = fd.FacetNormal(self.V.mesh())
        self.ls_ext.solve(self.rhs, deriv)
        fd.assemble(fd.inner(self.rhs, n) * v * fd.ds, tensor=out.fun)

    def get_params(self):
        """PETSc parameters to solve linear system."""
        params = {
            'ksp_rtol': 1e-12,
            'ksp_atol': 1e-12,
            'ksp_stol': 1e-12,
            'ksp_type': 'cg',
        }
        if self.direct_solve:
            params["pc_type"] = "cholesky"
            params["pc_factor_mat_solver_type"] = "mumps"
        else:
            params["pc_type"] = "hypre"
            params["pc_hypre_type"] = "boomeramg"
        return params
