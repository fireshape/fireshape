import fireshape.zoo as fsz
from firedrake import split, TestFunctions, inner, grad, dx, dot, div, \
    Constant, NonlinearVariationalProblem, NonlinearVariationalSolver, \
    ConvergenceError, tr, sym, sqrt, replace
from eikonal_solver import EikonalSolver
import numpy as np


class NavierStokesSolver(fsz.FluidSolver):

    def __init__(self, *args, porosity=Constant(0.0), **kwargs):
        self.porosity = porosity
        super().__init__(*args, **kwargs)
        self.problem = NonlinearVariationalProblem(self.F,
                                                   self.solution,
                                                   bcs=self.bcs)
        self.solver = NonlinearVariationalSolver(self.problem,
                                                 nullspace=self.nsp,
                                                 transpose_nullspace=self.nsp,
                                                 solver_parameters=self.params)

    def solve(self):
        super().solve()
        self.solver.solve()
        return self.solution

    def solve_by_continuation(self, start_nu=1e-2, steps=10, pre_solve_cb=None,
                              post_solve_cb=None,
                              zero_initial_guess=True):
        from math import log10
        nuspace = np.logspace(log10(start_nu), log10(self.nu.values()[0]),
                              steps, endpoint=True)
        if zero_initial_guess:
            self.solution *= 0
        for _nu in nuspace:
            self.nu.assign(_nu)
            try:
                if pre_solve_cb is not None:
                    pre_solve_cb(_nu)
                self.solve()
                if post_solve_cb is not None:
                    post_solve_cb(_nu)
            except ConvergenceError:
                print("Failed to converge for nu={}".format(_nu))
                raise

    def get_weak_form(self):
        (v, q) = TestFunctions(self.V)
        u = split(self.solution)[0]
        p = split(self.solution)[1]
        F = (
            self.nu * inner(grad(u), grad(v)) * dx
            + inner(dot(grad(u), u), v) * dx
            - p * div(v) * dx
            + div(u) * q * dx
            + self.porosity * inner(u, v) * dx
        )
        return F

    def get_parameters(self):
        snes_params = {
            "snes_monitor": False,
            "snes_max_it": 50,
            "snes_atol": 1e-10,
            "snes_linesearch_type": "l2"
        }
        if self.direct:
            ksp_params = {
                "ksp_monitor": False,
                "ksp_type": "fgmres",
                "mat_type": "aij",
                "pc_type": "lu",
                "pc_factor_mat_solver_type": "mumps",
                "ksp_atol": 1e-15,
            }
        else:
            raise NotImplementedError("This is not really working so far.")
        return {**snes_params, **ksp_params}

    def derivative_form(self, deformation):
        w = deformation
        u = self.solution.split()[0]
        p = self.solution.split()[1]
        v = self.solution_adj.split()[0]
        q = self.solution_adj.split()[1]
        nu = self.nu

        deriv = -inner(nu * grad(u) * grad(w),
                       grad(v)) * dx
        deriv -= inner(nu * grad(u),
                       grad(v) * grad(w)) * dx
        deriv -= inner(v, grad(u)*grad(w)*u) * dx
        deriv += tr(grad(v)*grad(w)) * p * dx
        deriv -= tr(grad(u)*grad(w)) * q * dx

        deriv += div(w) * inner(nu * grad(u), grad(v)) * dx
        deriv -= div(w) * inner(div(v), p) * dx
        deriv += div(w) * inner(div(u), q) * dx
        deriv += div(w) * inner(v, grad(u)*u) * dx
        deriv += div(w) * self.porosity * inner(v, u) * dx
        return deriv


class RANSMixingLengthSolver(NavierStokesSolver):

    def __init__(self, *args, dmax=Constant(1000.0), dshift=Constant(0.0),
                 **kwargs):
        self.kappa = 0.41
        self.dmax = dmax
        self.dshift = dshift
        super().__init__(*args, **kwargs)

    def get_weak_form(self):
        F = super().get_weak_form()
        self.eikonal_solver = EikonalSolver(self.mesh_m, self.noslip_bids)
        self.eikonal_solver.solve()
        u = split(self.solution)[0]
        v = split(F.arguments()[0])[0]
        F += self.nut() * inner(sym(grad(u)), grad(v)) * dx(domain=self.mesh_m)
        return F

    def nut(self):
        d = self.eikonal_solver.solution
        u = split(self.solution)[0]
        omega = sym(grad(u))

        def Min(a, b): return (a+b-abs(a-b))/Constant(2)

        d = Min(d, self.dmax) + self.dshift
        lmix = self.kappa * d
        # Since sqrt is not differentiable in 0, we have to add a small term
        # here as we get NaNs in our Jacobian otherwise.
        nut = (lmix**2) * sqrt(inner(omega, omega) + 1e-20)
        return nut

    def solve(self):
        self.eikonal_solver.solve()
        super().solve()

    def solve_adjoint(self, J):
        super().solve_adjoint(J)
        eikonal_j = replace(self.F, {self.F.arguments()[0]: self.solution_adj})
        self.eikonal_solver.solve_adjoint(eikonal_j)

    def derivative_form(self, deformation):
        w = deformation
        deriv = super().derivative_form(w)
        u = self.solution.split()[0]
        v = self.solution_adj.split()[0]
        d = self.eikonal_solver.solution

        def Min(a, b): return (a+b-abs(a-b))/Constant(2)
        d = Min(d, self.dmax) + self.dshift

        omega = sym(grad(u))
        lmix = (self.kappa**2) * (d**2)
        nut = lmix * sqrt(inner(omega, omega) + 1e-20)

        deriv -= lmix * inner(sym(grad(u)*grad(w)),
                              sym(grad(u)))/sqrt(inner(omega, omega)) \
            * inner(sym(grad(u)), sym(grad(v))) * dx
        deriv -= nut * inner(sym(grad(u)*grad(w)), grad(v)) * dx
        deriv -= nut * inner(sym(grad(u)), grad(v)*grad(w)) * dx

        deriv += nut * inner(sym(grad(u)), grad(v)) * div(w) * dx
        deriv += self.eikonal_solver.derivative_form(w)
        return deriv
