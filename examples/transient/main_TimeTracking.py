from firedrake import *
from fireshape import *
import ROL

class TimeTracking(PDEconstrainedObjective):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        mesh_m = self.Q.mesh_m
        x, y = SpatialCoordinate(mesh_m)

        self.J = 0
        self.File = File("u0.pvd")

        # target sol
        self.u_t= lambda t:  sin(pi*x)*sin(pi*y)*cos(t)
        f = lambda t: sin(pi*x)*sin(pi*y)*(2*pi**2*cos(t) - sin(t))

        V = FunctionSpace(mesh_m, "CG", 1)

        # incorrect initial guess
        u0 = Function(V)
        g = sin(y*pi)  # truncate at bdry
        perturbation = 0.05*sin(x*pi)*g**2
        u0.interpolate(sin(pi*x*(1+perturbation))*sin(pi*y))
        self.u0 = u0
        self.u = Function(V)
        self.u_old = Function(V)  # solution at previous time

        self.bcs = DirichletBC(V, 0, "on_boundary")
        dt = 0.1; self.dt = dt
        v = TestFunction(V)
        # explicit Euler method
        self.F = lambda t, u, u_old: inner((u-u_old)/dt,v)*dx(metadata={"quadrature_degree": 2}) + inner(grad(u_old), grad(v))*dx(metadata={"quadrature_degree": 2}) - inner(f(t), v)*dx(metadata={"quadrature_degree": 2})

    def solvePDE(self):
        self.J = 0; t = 0
        self.u.assign(self.u0)
        self.J += norm(self.u - self.u_t(t))**2

        for ii in range(10):
            self.u_old.assign(self.u)
            solve(self.F(t, self.u, self.u_old) == 0, self.u, bcs=self.bcs)
            t += self.dt
            self.J += norm(self.u - self.u_t(t))**2

    def value(self, x, tol):
        return self.J

    def cb():
        self.file.write(self.u0)

if __name__ == "__main__":
    # setup problem
    mesh = UnitSquareMesh(10, 10)
    Q = FeControlSpace(mesh)
    q = ControlVector(Q, LaplaceInnerProduct(Q, fixed_bids=[1, 2, 3, 4]))
    J = TimeTracking(Q)

    params_dict = {'Step': {'Type': 'Trust Region'},
                   'General': {'Secant': {'Type': 'Limited-Memory BFGS',
                                          'Maximum Storage': 25}},
                   'Status Test': {'Gradient Tolerance': 1e-6,
                                   'Step Tolerance': 1e-8,
                                   'Iteration Limit': 40}}

    # assemble and solve ROL optimization problem
    params = ROL.ParameterList(params_dict, "Parameters")
    problem = ROL.OptimizationProblem(J, q)
    solver = ROL.OptimizationSolver(problem, params)
    solver.solve()
