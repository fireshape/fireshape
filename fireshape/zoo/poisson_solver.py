import firedrake as fd
from ..pde_constraint import PdeConstraint

__all__ = ["PoissonSolver"]


class PoissonSolver(PdeConstraint):
    """Abstract class for fluid problems as PdeContraint."""
    def __init__(self, mesh_m, mini=False, direct=True,
                 dirichlet_bids=[], dirichlet_vals=[], rhs=fd.Constant(0.0)):
        super().__init__()
        self.mesh_m = mesh_m
        V = fd.FunctionSpace(mesh_m, "CG", 1)
        self.V = V
        self.bcs = []
        for (i, g) in zip(dirichlet_bids, dirichlet_vals):
            self.bcs.append(fd.DirichletBC(V, g, i))

        self.solution = fd.Function(V, name="State")
        self.solution_adj = fd.Function(V, name="Adjoint")

        v = fd.TestFunction(V)
        self.F = fd.inner(fd.grad(self.solution),
                          fd.grad(v)) * fd.dx - rhs * v * fd.dx
        self.f = rhs

        if len(self.bcs) > 0:
            self.nsp = None
        else:
            self.nsp = fd.VectorSpaceBasis(constant=True)

        self.params = {
            # 'snes_monitor': True,
            'snes_atol': 1e-10,
            'ksp_rtol': 1e-11,
            'ksp_atol': 1e-11,
            'ksp_stol': 1e-16,
            'ksp_type': 'cg',
            'pc_type': 'hypre',
            'pc_hypre_type': 'boomeramg'
            }

    def solve(self):
        super().solve()
        fd.solve(self.F == 0, self.solution, bcs=self.bcs,
                 nullspace=self.nsp, transpose_nullspace=self.nsp,
                 solver_parameters=self.params)

    def derivative_form(self, deformation):
        """Shape directional derivative of self.F wrt to w."""
        w = deformation
        u = self.solution
        v = self.solution_adj
        f = self.f

        deriv = (fd.inner(fd.grad(u), fd.grad(v)) - f * v) * fd.div(w) * fd.dx
        deriv -= fd.inner(fd.nabla_grad(w) * fd.grad(u), fd.grad(v)) * fd.dx
        deriv -= fd.inner(fd.grad(u), fd.nabla_grad(w) * fd.grad(v)) * fd.dx
        deriv -= fd.inner(fd.grad(f), v *  w) * fd.dx

        # deriv = fd.inner(fd.grad(u), fd.grad(v)) * fd.div(w) * fd.dx
        # deriv -= fd.inner(f, v) * fd.div(w) * fd.dx
        return deriv
