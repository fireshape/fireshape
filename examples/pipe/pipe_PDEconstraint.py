import firedrake as fd
import firedrake_adjoint as fda
from fireshape import PdeConstraint


class NavierStokesSolver(PdeConstraint):
    """Incompressible Navier-Stokes as PDE constraint."""

    def __init__(self, mesh_m, viscosity):
        super().__init__()
        self.mesh_m = mesh_m

        # Setup problem
        self.V = fd.VectorFunctionSpace(self.mesh_m, "CG", 2) \
            * fd.FunctionSpace(self.mesh_m, "CG", 1)

        # Preallocate solution variables for state equation
        self.solution = fda.Function(self.V, name="State")
        self.testfunction = fd.TestFunction(self.V)
        #### Preallocate solution variables for state and adjoint equations
        #self.solution_adj = fd.Function(self.V, name="Adjoint")

        # Define viscosity parameter
        self.viscosity = viscosity
        nu = self.viscosity

        # Weak form of incompressible Navier-Stokes equations
        z = self.solution
        u, p = fd.split(z)
        test = self.testfunction
        v, q = fd.split(test)
        self.F = nu*fd.inner(fd.grad(u), fd.grad(v))*fd.dx - p*fd.div(v)*fd.dx \
            + fd.inner(fd.dot(fd.grad(u), u), v)*fd.dx + fd.div(u)*q*fd.dx

        # Dirichlet Boundary conditions
        X = fd.SpatialCoordinate(self.mesh_m)
        dim = self.mesh_m.topological_dimension()
        if dim == 2:
            uin = 6 * fd.as_vector([(1-X[1])*X[1], 0])
        elif dim == 3:
            r = fd.sqrt(X[0]**2+X[1]**2)
            uin = fd.as_vector([0, 0, 1-(2*r)**2])
        else:
            raise NotImplementedError
        self.bcs = [fda.DirichletBC(self.V.sub(0), 0., [3, 4]),
                    fda.DirichletBC(self.V.sub(0), uin, 1)]

        # PDE-solver parameters
        self.nsp = None
        self.params = {"mat_type": "aij", "pc_type": "lu",
                       "pc_factor_mat_solver_type": "mumps"}

        problem = fda.NonlinearVariationalProblem(
            self.F, self.solution, bcs=self.bcs)
        self.solver = fd.NonlinearVariationalSolver(
            problem, solver_parameters=self.params)

    def solve(self):
        super().solve()
        self.solver.solve()
