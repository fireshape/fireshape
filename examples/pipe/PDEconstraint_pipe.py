import firedrake as fd
from fireshape import PdeConstraint


class NavierStokesSolver(PdeConstraint):
    """Incompressible Navier-Stokes as PDE constraint."""

    def __init__(self, mesh_m, viscosity):
        super().__init__()
        self.mesh_m = mesh_m
        self.failed_to_solve = False  # when self.solver.solve() fail

        # Setup problem, Taylor-Hood finite elements
        self.V = fd.VectorFunctionSpace(self.mesh_m, "CG", 2) \
            * fd.FunctionSpace(self.mesh_m, "CG", 1)

        # Preallocate solution variables for state equation
        self.solution = fd.Function(self.V, name="State")
        self.testfunction = fd.TestFunction(self.V)

        # Define viscosity parameter
        self.viscosity = viscosity

        # Weak form of incompressible Navier-Stokes equations
        z = self.solution
        u, p = fd.split(z)
        test = self.testfunction
        v, q = fd.split(test)
        nu = self.viscosity  # shorten notation
        self.F = nu*fd.inner(fd.grad(u), fd.grad(v))*fd.dx - p*fd.div(v)*fd.dx\
            + fd.inner(fd.dot(fd.grad(u), u), v)*fd.dx + fd.div(u)*q*fd.dx

        # Dirichlet Boundary conditions
        X = fd.SpatialCoordinate(self.mesh_m)
        dim = self.mesh_m.topological_dimension
        if dim == 2:
            uin = 4 * fd.as_vector([(1-X[1])*X[1], 0])
        elif dim == 3:
            rsq = X[0]**2+X[1]**2  # squared radius = 0.5**2 = 1/4
            uin = fd.as_vector([0, 0, 1-4*rsq])
        else:
            raise NotImplementedError
        self.bcs = [fd.DirichletBC(self.V.sub(0), 0., [12, 13]),
                    fd.DirichletBC(self.V.sub(0), uin, [10])]

        # PDE-solver parameters
        self.nsp = None
        self.params = {
            "snes_max_it": 10, "mat_type": "aij", "pc_type": "lu",
            "pc_factor_mat_solver_type": "superlu_dist",
            # "snes_monitor": None, "ksp_monitor": None,
        }

    def solve(self):
        super().solve()
        self.failed_to_solve = False
        u_old = self.solution.copy(deepcopy=True)
        try:
            fd.solve(self.F == 0, self.solution, bcs=self.bcs,
                     solver_parameters=self.params)
        except fd.ConvergenceError:
            self.failed_to_solve = True
            self.solution.assign(u_old)


if __name__ == "__main__":
    mesh = fd.Mesh("pipe.msh")
    if mesh.topological_dimension == 2:  # in 2D
        viscosity = fd.Constant(1./400.)
    elif mesh.topological_dimension == 3:  # in 3D
        viscosity = fd.Constant(1/10.)  # simpler problem in 3D
    else:
        raise NotImplementedError
    e = NavierStokesSolver(mesh, viscosity)
    e.solve()
    print(e.failed_to_solve)
    out = fd.File("temp_PDEConstrained_u.pvd")
    out.write(e.solution.split()[0])
