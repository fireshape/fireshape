from firedrake import *
from fireshape import *
import ROL


class TargetHeat(PDEconstrainedObjective):
    """L2 misfit function constrained to the heat equation."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.mesh = self.Q.mesh_m

        # Setup problem
        V = FunctionSpace(self.mesh_m, "CG", 1)
        self.u = Function(V, name="Temperature")
        self.u_ = Function(V)
        self.u0 = Function(V)  #initial condition

        # Time discretization parameters
        self.N = 10  # number of time steps
        self.dt = (pi/2)/self.N  # time step length

        # target temperature profile, exact solution is a ball of radius 0.8
        # centered at (0.5,0.5,0.5)
        (x, y, z) = SpatialCoordinate(self.mesh_m)
        self.u_t = 0.64 - (x - 0.5)**2 - (y - 0.5)**2 - (z - 0.5)**2

        # Weak form of implicit Euler applied to Heat equation
        u = self.u
        u_ = self.u_
        v = TestFunction(V)
        self.a = Constant(6 * sin(self.dt))
        self.b = Constant(cos(self.dt))
        f = self.a + self.b * self.u_t  # manufactured source term
        F = ((u - u_)*v/self.dt + inner(grad(u), grad(v)) - f * v) * dx
        bcs = DirichletBC(V, 0., "on_boundary")
        stateproblem = NonlinearVariationalProblem(F, self.u, bcs=bcs)
        self.solver = NonlinearVariationalSolver(stateproblem)

    def compute_temperature(self, name=None):
        """
        Solve heat equation. If input name is passed, store
        temperature time evolution in corresponding file.
        """
        # assign initial condition
        self.u_.assign(self.u0)
        self.u.assign(self.u0)

        if name is not None:
            print("Storing the "+name+" temperature evolution.")
            out = VTKFile(name+"_temperature_evolution.pvd")
            out.write(self.u, time=0)

        for ii in range(self.N):
            # update source term coefficients
            t = (ii + 1)*self.dt
            self.a.assign(6 * sin(t))
            self.b.assign(cos(t))
            # perform time step
            self.solver.solve()
            self.u_.assign(self.u)
            if name is not None:
                out.write(self.u, time=t)

    def objective_value(self):
        self.compute_temperature()
        return assemble((self.u - self.u_t)**2 * dx)

# Select initial guess, control space, and inner product
mesh = UnitBallMesh(refinement_level=3)
Q = FeControlSpace(mesh)
IP = H1InnerProduct(Q)
q = ControlVector(Q, IP)

# Instantiate objective function J
out = VTKFile("domain.pvd")
J = TargetHeat(Q, cb=lambda: out.write(Q.mesh_m.coordinates))
J.compute_temperature("initial")

# Select the optimization algorithm and solve the problem
pd = {'Step': {'Type': 'Trust Region'},
      'General':  {'Secant': {'Type': 'Limited-Memory BFGS',
                                       'Maximum Storage': 25}},
       'Status Test': {'Gradient Tolerance': 1e-3,
                       'Step Tolerance': 1e-8,
                       'Iteration Limit': 30}}
params = ROL.ParameterList(pd, "Parameters")
problem = ROL.OptimizationProblem(J, q)
solver = ROL.OptimizationSolver(problem, params)
solver.solve()
J.compute_temperature("final")
