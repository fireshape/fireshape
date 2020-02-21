import firedrake as fd
from ..pde_constraint import PdeConstraint

__all__ = ["StokesSolver"]


class FluidSolver(PdeConstraint):
    """Abstract class for fluid problems as PdeContraint."""

    def __init__(self, mesh_m, mini=False, direct=True,
                 inflow_bids=[], inflow_expr=None,
                 noslip_bids=[], nu=1.0):
        """
        Instantiate a FluidSolver.

        Inputs:
            mesh_m: type fd.Mesh
            mini: type bool, set to true to use MINI elments
            direct: type bool, set to True to use direct solver
            inflow_bids: type list (of ints), list of inflow bdries
            inflow_expr: type ???UFL??, UFL formula for inflow bdry conditions
            noslip_bids: typ list (of ints), list of bdries with homogeneous
                         Dirichlet bdry condition for velocity
            nu: type float, viscosity
        """
        super().__init__()
        self.mesh_m = mesh_m
        self.mini = mini
        self.direct = direct
        self.inflow_bids = inflow_bids
        self.inflow_expr = inflow_expr
        self.noslip_bids = noslip_bids
        self.nu = fd.Constant(nu)

        # Setup problem
        self.V = self.get_functionspace()

        # Manage boundary conditions
        # this only works after a functionspace on the mesh has been declared..
        all_bids = list(self.mesh_m.topology.exterior_facets.unique_markers)
        for bid in inflow_bids + noslip_bids:
            all_bids.remove(bid)
        self.outflow_bids = all_bids

        # Preallocate solution variables for state and adjoint equations
        self.solution = fd.Function(self.V, name="State")

        self.F = self.get_weak_form()
        self.bcs = self.get_boundary_conditions()
        self.nsp = self.get_nullspace()
        self.params = self.get_parameters()
        # problem = fd.NonlinearVariationalProblem(self.F, self.solution,
        #                                           bcs=self.bcs,)
        # self.solver = fd.NonlinearVariationalSolver(
        #     problem, solver_parameters=self.params, nullspace=self.nsp)

    def solve(self):
        super().solve()
        # self.solver.solve()
        fd.solve(self.F == 0, self.solution, bcs=self.bcs, solver_parameters=self.params, nullspace=self.nsp)

    def get_functionspace(self):
        """Construct trial/test space for state and adjoint equations."""
        if self.mini:
            # MINI elements
            mini = fd.FiniteElement("CG", fd.triangle, 1) \
                + fd.FiniteElement("B", fd.triangle, 3)
            Vvel = fd.VectorFunctionSpace(self.mesh_m, mini)
        else:
            # P2/P1 Taylor-Hood elements
            Vvel = fd.VectorFunctionSpace(self.mesh_m, "Lagrange", 2)
        Vpres = fd.FunctionSpace(self.mesh_m, "CG", 1)
        return Vvel * Vpres

    def get_boundary_conditions(self):
        """Impose Dirichlet boundary conditions."""
        dim = self.mesh_m.cell_dimension()
        if dim == 2:
            zerovector = fd.Constant((0.0, 0.0))
        elif dim == 3:
            zerovector = fd.Constant((0.0, 0.0, 0.0))

        bcs = []
        if len(self.inflow_bids) is not None:
            bcs.append(fd.DirichletBC(self.V.sub(0), self.inflow_expr,
                                       self.inflow_bids))
        if len(self.noslip_bids) > 0:
            bcs.append(fd.DirichletBC(self.V.sub(0), zerovector,
                                       self.noslip_bids))
        return bcs

    def get_nullspace(self):
        """Specify nullspace of state/adjoint equation."""

        if len(self.outflow_bids) > 0:
            # If the pressure is fixed (anywhere) by a Dirichlet bc, nsp = None
            nsp = None
        else:
            nsp = fd.MixedVectorSpaceBasis(
                self.V, [self.V.sub(0), fd.VectorSpaceBasis(constant=True)])
        return nsp


class StokesSolver(FluidSolver):
    """Implementation of Stokes' problem as PdeConstraint."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def get_weak_form(self):
        (v, q) = fd.TestFunctions(self.V)
        (u, p) = fd.split(self.solution)
        F = self.nu * fd.inner(fd.grad(u), fd.grad(v)) * fd.dx \
            - p * fd.div(v) * fd.dx \
            + fd.div(u) * q * fd.dx \
            + fd.inner(fd.Constant((0., 0.)), v) * fd.dx
        return F

    def get_parameters(self):
        if self.direct:
            ksp_params = {
                # "ksp_monitor": shopt_parameters['verbose_state_solver'],
                "ksp_type": "fgmres",
                "mat_type": "aij",
                "pc_type": "lu",
                "pc_factor_mat_solver_type": "mumps",
                "ksp_atol": 1e-15,
            }
        else:
            # reuse initial guess!
            raise NotImplementedError("Iterative solver has not been "
                                      "implemented.")
        return ksp_params
