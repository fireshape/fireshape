import firedrake as fd
from ..pde_constraint import PdeConstraint

__all__ = ["PoissonSolver"]

class PoissonSolver(PdeConstraint):
    """Class for fullH1 (good Helmholtz, hom NeumannBC) problems as PdeContraint."""
    def __init__(self, mesh_m): 
        super().__init__()
        self.mesh_m = mesh_m

        # Setup problem
        self.V = fd.FunctionSpace(self.mesh_m, "CG", 1)

        # Preallocate solution variables for state and adjoint equations
        self.solution = fd.Function(self.V, name="State")
        self.solution_adj = fd.Function(self.V, name="Adjoint")

        self.f = fd.Constant(4.)
        #u, v = fd.TrialFunction(self.V), fd.TestFunction(self.V)
        v = fd.TestFunction(self.V)
        self.F = fd.inner(fd.grad(self.solution), fd.grad(v)) * fd.dx  - self.f*v*fd.dx
        self.bcs = fd.DirichletBC(self.V, 0., "on_boundary")
        self.nsp = None
        self.params = {
                # "ksp_monitor": shopt_parameters['verbose_state_solver'],
                "ksp_type": "fgmres",
                "mat_type": "aij",
                "pc_type": "lu",
                "pc_factor_mat_solver_package": "mumps",
                "ksp_atol": 1e-15,
                      }


    def solve(self):
        super().solve()
        fd.solve(self.F == 0, self.solution, bcs=self.bcs,
                 solver_parameters=self.params)
        return self.solution

    def derivative_form(self, deformation):
        """Shape directional derivative of self.F wrt to w."""
        w = deformation
        u = self.solution
        p = self.solution_adj
        f = self.f
        Dw = fd.nabla_grad(w)

        deriv = fd.div(w) * (fd.inner(fd.grad(u), fd.grad(p)) - f * p) * fd.dx
        deriv -= fd.inner(Dw * fd.grad(u), fd.grad(p)) * fd.dx
        deriv -= fd.inner(fd.grad(u), Dw * fd.grad(p)) * fd.dx
        #deriv -= fd.inner(fd.grad(f), w) * p * fd.dx
        return deriv
