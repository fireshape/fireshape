import firedrake as fd
from fireshape import PdeConstraint

class NavierStokesSolver(PdeConstraint):
    """Incompressible Navier-Stokes as PDE constraint."""
    def __init__(self, mesh_m):
        super().__init__()
        self.mesh_m = mesh_m

        # Setup problem
        self.V = fd.VectorFunctionSpace(self.mesh_m, "CG", 2) \
                 * fd.FunctionSpace(self.mesh_m, "CG", 1)

        # Preallocate solution variables for state and adjoint equations
        self.solution = fd.Function(self.V, name="State")
        self.testfunction = fd.TestFunction(self.V)
        self.solution_adj = fd.Function(self.V, name="Adjoint")

        # Define viscosity parameter
        self.viscosity = 1./400.
        nu = self.viscosity

        # Weak form of Poisson problem
        z = self.solution
        u, p = fd.split(z)
        test = self.testfunction
        v, q = fd.split(test)

        self.F = nu*fd.inner(fd.grad(u), fd.grad(v))*fd.dx - p*fd.div(v)*fd.dx \
                 + fd.inner(fd.dot(fd.grad(u), u), v)*fd.dx + fd.div(u)*q*fd.dx
        X = fd.SpatialCoordinate(self.mesh_m)
        uin = 6 * fd.as_vector([(1-X[1])*X[1], 0])
        self.bcs = [fd.DirichletBC(self.V.sub(0), 0., [3, 4]),
                    fd.DirichletBC(self.V.sub(0), uin, 1)]

        # PDE-solver parameters
        self.nsp = None
        self.params = {"mat_type": "aij", "pc_type": "lu",
                "pc_factor_mat_solver_package": "mumps"}

        stateproblem = fd.NonlinearVariationalProblem(self.F, self.solution, bcs=self.bcs)
        self.stateproblem = fd.NonlinearVariationalSolver(stateproblem, solver_parameters=self.params)
