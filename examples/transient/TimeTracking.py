from firedrake import *
from fireshape import *

def TimeTracking(PDEconstrainedObjective):

    def __init__(self, *args, **kwargs):

        mesh_m = self.Q.mesh_m
        x, y = SpatialCoordinates(mesh_m)

        self.J = 0
        self.File = File("u0.pvd")

        # target sol
        self.u_t= lambda t:  sin(pi*x)*sin(pi*y)*cos(t)
        f = lambda t: sin(pi*x)*sin(pi*y)*(2*pi**2*cos(t) - sin(t))

        V = FunctionSpace(mesh, "CG", 1)

        # incorrect initial guess
        u0 = Function(V)
        g = sin(y*pi)  # truncate at bdry
        perturbation = 0.05*sin(x*pi)*g**2
        u0 = interpolate(sin(pi*x*(1+pertubation))*sin(pi*y))
        self.u0 = u0
        self.u = Function(V)
        self.u_old = Function(V)  # solution at previous time

        self.bcs = DirichletBC(V, 0, "on_boundary")
        h = 0.1; self.h = h
        v = TestFunction(V)
        # explicit Euler method
        F = lambda t, u, u_old: inner((u-u_old)/h,v)*dx + inner(grad(u_old), grad(v))*dx - inner(f(t), v)*dx

    def solvePDE(self):
        self.J = 0; t = 0
        self.u.assign(self.u0)
        self.J += norm(self.u - self.u_t(t))**2

        for ii in range(10):
            self.u_old.assing(self.u0)
            solve(F(t, self.u, self.u_old) == 0, self.u, bcs=self.bcs)
            t += self.dt
            self.J += norm(self.u - self.u_t(t))**2

    def value(self, x, tol):
        return self.J

    def cb():
        self.file.write(self.u0)

