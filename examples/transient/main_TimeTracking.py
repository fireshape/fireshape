from firedrake import *
from fireshape import *
import ROL

class TimeTracking(PDEconstrainedObjective):
    """
    L1-L2 misfit functional for time-dependent problem constrained
    to the heat equation. This toy problem is solved by optimizing the
    mesh so that a finite element function defined on it becomes
    the right initial value of the heat equation.

    The value of the output functional is computed along
    the time stepping.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # container for functional value
        self.J = 0

        # target solution, which also specifies rhs and DirBC
        mesh_m = self.Q.mesh_m
        x, y = SpatialCoordinate(mesh_m)
        self.u_t= lambda t:  sin(pi*x)*sin(pi*y)*cos(t)
        f = lambda t: sin(pi*x)*sin(pi*y)*(2*pi**2*cos(t) - sin(t))

        # perturped initial guess to be fixed by shape optimization
        V = FunctionSpace(mesh_m, "CG", 1)
        self.u0 = Function(V)
        perturbation = 0.25*sin(x*pi)*sin(y*pi)**2
        self.u0.interpolate(sin(pi*x*(1+perturbation))*sin(pi*y))

        # define self.cb, which is always called after self.solvePDE
        self.File = File("u0.pvd")
        self.cb = lambda : self.File.write(self.u0)

        # heat equation discreized with implicit Euler
        self.u = Function(V)
        self.u_old = Function(V)  # solution at previous time
        self.bcs = DirichletBC(V, 0, "on_boundary")
        self.dx = dx(metadata={"quadrature_degree": 1})
        self.dt = 0.125
        v = TestFunction(V)
        self.F = lambda t, u, u_old: inner((u-u_old)/self.dt,v)*self.dx \
                                     + inner(grad(u), grad(v))*self.dx \
                                     - inner(f(t+self.dt), v)*self.dx

    def solvePDE(self):
        """Solve the heat equation and evaluate the objective function."""
        self.J = 0; t = 0
        self.u.assign(self.u0)
        self.J += assemble(self.dt*(self.u - self.u_t(t))**2*self.dx)

        for ii in range(10):
            self.u_old.assign(self.u)
            solve(self.F(t, self.u, self.u_old) == 0, self.u, bcs=self.bcs)
            t += self.dt
            self.J += assemble(self.dt*(self.u - self.u_t(t))**2*self.dx)

    def objective_value(self):
        """Return the value of the objective function."""
        return self.J


if __name__ == "__main__":

    # setup problem
    mesh = UnitSquareMesh(20, 20)
    Q = FeControlSpace(mesh)
    q = ControlVector(Q, LaplaceInnerProduct(Q, fixed_bids=[1, 2, 3, 4]))
    J = TimeTracking(Q)

    params_dict = {'Step': {'Type': 'Trust Region'},
                   'General': {'Secant': {'Type': 'Limited-Memory BFGS',
                                          'Maximum Storage': 25}},
                   'Status Test': {'Gradient Tolerance': 1e-3,
                                   'Step Tolerance': 1e-8,
                                   'Iteration Limit': 20}}

    # assemble and solve ROL optimization problem
    params = ROL.ParameterList(params_dict, "Parameters")
    problem = ROL.OptimizationProblem(J, q)
    solver = ROL.OptimizationSolver(problem, params)
    solver.solve()
