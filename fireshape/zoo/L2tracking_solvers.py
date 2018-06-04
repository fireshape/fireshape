import firedrake as fd
from ..pde_constraint import PdeConstraint

__all__ = ["PoissonSolver"]

class PoissonSolver(PdeConstraint):
    """Class for Poisson problems as PdeContraint."""
    def __init__(self, mesh_m): 
        """
        Inputs:
            mesh_m: type fd.Mesh
        """
        super().__init__()
        self.mesh_m = mesh_m
        self.direct = True

        # Setup problem
        self.V = FunctionSpace(self.mesh_m, "CG", 1)

        # Preallocate solution variables for state and adjoint equations
        self.solution = fd.Function(self.V, name="State")
        self.solution_adj = fd.Function(self.V, name="Adjoint")

        u, v = Function(self.V), TestFuntion(self.v)
        self.F = inner(grad(u), grad(v)) * dx  + u*v*dx - Constant(4.)*v*dx
        self.bcs = []
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
        fd.solve(fd.lhs(self.F) == fd.rhs(self.F), self.solution, bcs=self.bcs,
                 solver_parameters=self.params)
        return self.solution

    def derivative_form(self, deformation):
        """Shape directional derivative of self.F wrt to w."""
        w = deformation
        u = self.solution
        p = self.solution_adj

        deriv = -fd.inner(fd.grad(u), (fd.grad(w)+ transpose(fd.grad(w))*fd.grad(v)) * fd.dx
        deriv += fd.div(w) * fd.inner(fd.grad(u), fd.grad(v)) * fd.dx
        deriv += fd.div(w) * u * p * fd.dx
        deriv -= fd.div(w) * Constant(4.) * q * fd.dx
        return deriv
