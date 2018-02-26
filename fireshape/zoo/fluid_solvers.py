import firedrake as fd
from ..pde_constraint import PdeConstraint

__all__ = ["StokesSolver"]

class FluidSolver(PdeConstraint):

    def __init__(self, m_mesh, mini=False, direct=True,
                 inflow_bids=[], inflow_expr=None,
                 noslip_bids=[], nu=1.0):

        super().__init__()
        self.m_mesh = m_mesh
        self.mini = mini
        self.direct = direct
        self.inflow_bids = inflow_bids
        self.inflow_expr = inflow_expr
        self.noslip_bids = noslip_bids
        self.nu = fd.Constant(nu)

        """ Setup problem """
        self.V = self.get_functionspace()

        # this only works after a functionspace on the mesh has been declared..
        all_bids = list(self.m_mesh.topology.exterior_facets.unique_markers)
        for bid in inflow_bids + noslip_bids:
            all_bids.remove(bid)
        self.outflow_bids = all_bids

        self.solution = fd.Function(self.V, name="State")
        self.solution_adj = fd.Function(self.V, name="Adjoint")

        self.F = self.get_weak_form()
        self.bcs = self.get_boundary_conditions()
        self.nsp = self.get_nullspace()
        self.params = self.get_parameters()

    def get_functionspace(self):
        if self.mini:
            mini = fd.FiniteElement("CG", fd.triangle, 1) \
                + fd.FiniteElement("B", fd.triangle, 3)
            Vvel = fd.VectorFunctionSpace(self.m_mesh, mini)
        else:
            Vvel = fd.VectorFunctionSpace(self.m_mesh, "Lagrange", 2)
        Vpres = fd.FunctionSpace(self.m_mesh, "CG", 1)
        return Vvel * Vpres

    def get_boundary_conditions(self):
        dim = self.m_mesh.cell_dimension()
        if dim == 2:
            zerovector = fd.Constant((0.0, 0.0))
        elif dim == 3:
            zerovector = fd.Constant((0.0, 0.0, 0.0))

        bcs = []
        if len(self.inflow_bids) is not None:
            bcs.append(fd.DirichletBC(self.V.sub(0), self.inflow_expr,
                                   self.inflow_bids))
        if len(self.noslip_bids)>0:
            bcs.append(fd.DirichletBC(self.V.sub(0), zerovector,
                                   self.noslip_bids))
        return bcs

    def get_nullspace(self):
        # check if there is any condition fixing the pressure
        if len(self.outflow_bids) > 0:
            nsp = None
        else:
            nsp = fd.MixedVectorSpaceBasis(
                self.V, [self.V.sub(0), fd.VectorSpaceBasis(constant=True)])
        return nsp


class StokesSolver(FluidSolver):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def solve(self):
        super().solve()
        fd.solve(fd.lhs(self.F) == fd.rhs(self.F), self.solution, bcs=self.bcs,
              nullspace=self.nsp, transpose_nullspace=self.nsp,
              solver_parameters=self.params)
        return self.solution

    def get_weak_form(self):
        (v, q) = fd.TestFunctions(self.V)
        (u, p) = fd.TrialFunctions(self.V)
        F = (
            self.nu * fd.inner(fd.grad(u), fd.grad(v)) * fd.dx
            - p * fd.div(v) * fd.dx
            + fd.div(u) * q * fd.dx
            + fd.inner(fd.Constant((0., 0.)), v) * fd.dx
        )
        return F

    def get_parameters(self):
        if self.direct:
            ksp_params = {
                # "ksp_monitor": shopt_parameters['verbose_state_solver'],
                "ksp_type": "fgmres",
                "mat_type": "aij",
                "pc_type": "lu",
                "pc_factor_mat_solver_package": "mumps",
                "ksp_atol": 1e-15,
            }
        else:
            # reuse initial guess!
            raise NotImplementedError("Iterative solver has not been "
                                      "implemented.")
        return ksp_params

    def derivative_form(self, deformation):
        w = deformation
        u = self.solution.split()[0]
        p = self.solution.split()[1]
        v = self.solution_adj.split()[0]
        q = self.solution_adj.split()[1]
        nu = self.nu

        deriv = -fd.inner(nu * fd.grad(u) * fd.grad(w),
                       fd.grad(v)) * fd.dx
        deriv -= fd.inner(nu * fd.grad(u),
                       fd.grad(v) * fd.grad(w)) * fd.dx
        deriv += fd.tr(fd.grad(v)*fd.grad(w)) * p * fd.dx
        deriv -= fd.tr(fd.grad(u)*fd.grad(w)) * q * fd.dx

        deriv += fd.div(w) * fd.inner(nu * fd.grad(u), fd.grad(v)) * fd.dx
        deriv -= fd.div(w) * fd.inner(fd.div(v), p) * fd.dx
        deriv += fd.div(w) * fd.inner(fd.div(u), q) * fd.dx
        return deriv
