from firedrake import *
from fireshape import *
import fireshape.zoo as fsz
import ROL


class EnergyDissipation(PDEconstrainedObjective):
    """
    Energy dissipation due to viscosity in Navier-Stokes fluid.
    """
    def __init__(self, *args, viscosity, **kwargs):
        super().__init__(*args, **kwargs)
        mesh = self.Q.mesh_m

        # Setup problem, Taylor-Hood finite elements
        V = VectorFunctionSpace(mesh, "CG", 2) \
            * FunctionSpace(mesh, "CG", 1)

        # Preallocate solution variables for state equation
        self.solution = Function(V, name="State")
        testfunction = TestFunction(V)

        # Define viscosity parameter
        self.viscosity = viscosity

        # Setup problem
        z = self.solution
        u, p = split(z)
        v, q = split(testfunction)
        nu = self.viscosity  # shorten notation
        F = (nu*inner(grad(u), grad(v))*dx - p*div(v)*dx
             + inner(dot(grad(u), u), v)*dx + div(u)*q*dx)

        # Dirichlet Boundary conditions
        X = SpatialCoordinate(mesh)
        uin = 4 * as_vector([(1-X[1])*X[1], 0])
        bcs = [DirichletBC(V.sub(0), 0., [12, 13]),
               DirichletBC(V.sub(0), uin, [10])]

        # PDE-solver parameters
        # params = {
        #    "snes_max_it": 20, "mat_type": "aij", "pc_type": "lu",
        #    "pc_factor_mat_solver_type": "superlu_dist",
        #     "snes_monitor": None, "ksp_monitor": None,
        # }
        pms = {
            "mat_type": "aij",  # Matrix type (e.g., sparse matrix format)
            "snes_type": "newtonls",  # Use Newton's method with line search
            "ksp_type": "gmres",  # Direct solver for the linear system
            "pc_type": "lu",  # Use LU decomposition for preconditioning
            "pc_factor_mat_solver_type": "superlu_dist",
            # "snes_converged_reason": "",  # Print convergence reason
            # "snes_monitor": "",  # Monitor iterations during the solve
            "ksp_rtol": 1.0e-10,
            "snes_rtol": 1.0e-10,  # desired relative tolerance for SNES here
            "snes_atol": 1.0e-10,  # absolute tolerance for SNES
            "snes_max_it": 50,  # And the maximum number of iterations
            # "ksp_converged_reason":"",
            # "ksp_monitor":""
        }

        prb = NonlinearVariationalProblem(F, self.solution, bcs=bcs)
        self.solver = NonlinearVariationalSolver(prb, solver_parameters=pms)

    def compute_velocity(self, name=None):
        """
        Solve Navier Stokes equations. If input name is passed,
        store velocity and pressure in corresponding file.
        """
        # assign initial condition
        self.solver.solve()

        if name is not None:
            print("Storing the " + name + " fluid.")
            u, p = self.solution.subfunctions
            u.rename("velocity")
            p.rename("pressure")
            VTKFile(name + "_fluid.pvd").write(u, p)

    def objective_value(self):
        self.compute_velocity()
        u, p = split(self.solution)
        nu = self.viscosity
        return assemble(nu * inner(grad(u), grad(u)) * dx)


if __name__ == "__main__":
    # setup problem
    mesh = Mesh("pipe.msh")
    Q = FeControlSpace(mesh)
    IP = H1InnerProduct(Q, fixed_bids=[10, 11, 12])
    q = ControlVector(Q, IP)

    # setup PDE constraint
    viscosity = Constant(1./100.)

    # save state variable evolution in file u2.pvd or u3.pvd
    out = VTKFile("domain.pvd")
    cb = lambda: out.write(Q.mesh_m.coordinates)

    # create PDEconstrained objective functional
    J_ = EnergyDissipation(Q, cb=cb, viscosity=viscosity)
    J_.compute_velocity("initial")

    # add regularization to improve mesh quality
    Jq = fsz.MoYoSpectralConstraint(100, Constant(0.5), Q)
    J = J_ + Jq

    # Set up volume constraint
    vol = fsz.VolumeFunctional(Q)
    initial_vol = vol.value(q, None)
    econ = EqualityConstraint([vol], target_value=[initial_vol])
    emul = ROL.StdVector(1)

    # ROL parameters
    params_dict = {
        'General': {'Print Verbosity': 0,  # set to 1 to understand output
                    'Secant': {'Type': 'Limited-Memory BFGS',
                               'Maximum Storage': 10}},
        'Step': {'Type': 'Augmented Lagrangian',
                 'Augmented Lagrangian':
                 {'Subproblem Step Type': 'Trust Region',
                  'Print Intermediate Optimization History': False,
                  'Subproblem Iteration Limit': 10}},
        'Status Test': {'Gradient Tolerance': 1e-1,
                        'Step Tolerance': 1e-4,
                        'Constraint Tolerance': 1e-1,
                        'Iteration Limit': 30}}
    params = ROL.ParameterList(params_dict, "Parameters")
    problem = ROL.OptimizationProblem(J, q, econ=econ, emul=emul)
    solver = ROL.OptimizationSolver(problem, params)
    solver.solve()
    J_.compute_velocity("final")
