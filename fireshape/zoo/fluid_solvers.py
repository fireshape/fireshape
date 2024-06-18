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
        fd.solve(self.F == 0, self.solution, bcs=self.bcs,
                 solver_parameters=self.params, nullspace=self.nsp)

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
            ksp_params = {
            # First up we select the unassembled matrix type::

                "mat_type": "matfree",

            # Now we configure the solver, using GMRES using the diagonal part of
            # the Schur complement factorisation to approximate the inverse.  We'll
            # also monitor the convergence of the residual, and ask PETSc to view
            # the configured Krylov solver object.::

                "ksp_type": "gmres",
                # "ksp_monitor_true_residual": None,
                # "ksp_view": None,
                "pc_type": "fieldsplit",
                "pc_fieldsplit_type": "schur",
                "pc_fieldsplit_schur_fact_type": "diag",

            # Next we configure the solvers for the blocks.  For the velocity block,
            # we use an :class:`.AssembledPC` and approximate the inverse of the
            # vector laplacian using a single multigrid V-cycle.::

                "fieldsplit_0_ksp_type": "preonly",
                "fieldsplit_0_pc_type": "python",
                "fieldsplit_0_pc_python_type": "firedrake.AssembledPC",
                "fieldsplit_0_assembled_pc_type": "lu",

            # For the Schur complement block, we approximate the inverse of the
            # schur complement with a pressure mass inverse.  For constant viscosity
            # this works well.  For variable, but low-contrast viscosity, one should
            # use a viscosity-weighted mass-matrix.  This is achievable by passing a
            # dictionary with "mu" associated with the viscosity into solve.  The
            # MassInvPC will choose a default value of 1.0 if not set.  For high viscosity
            # contrasts, this preconditioner is mesh-dependent and should be replaced
            # by some form of approximate commutator.::

                "fieldsplit_1_ksp_type": "preonly",
                "fieldsplit_1_pc_type": "python",
                "fieldsplit_1_pc_python_type": "firedrake.MassInvPC",

            # The mass inverse is dense, and therefore approximated with an incomplete
            # LU factorization, which we configure now::

                "fieldsplit_1_Mp_mat_type": "aij",
                "fieldsplit_1_Mp_pc_type": "ilu"
            }

            # reuse initial guess!
            # raise NotImplementedError("Iterative solver has not been "
            #                           "implemented.")
        return ksp_params
