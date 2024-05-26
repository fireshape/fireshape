import firedrake as fd
import fireshape as fs
import petsc4py.PETSc as PETSc


class TimeTracking(fs.PDEconstrainedObjective):
    """
    L1-L2 misfit functional for time-dependent problem constrained
    to the heat equation. This toy problem is solved by optimizing the
    mesh interior so that a finite element function defined on it
    becomes the right initial value of the heat equation.

    The value of the output functional is computed along
    the time stepping.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # container for functional value
        self.J = 0

        # target solution, which also specifies rhs
        # leads to homogeneous DirichletBC
        mesh_m = self.Q.mesh_m
        x, y = fd.SpatialCoordinate(mesh_m)
        sin = fd.sin
        cos = fd.cos
        pi = fd.pi
        self.u_t = lambda t: sin(pi*x)*sin(pi*y)*cos(t)
        self.f = lambda t: sin(pi*x)*sin(pi*y)*(2*pi**2*cos(t) - sin(t))

        # perturbed initial guess to be fixed by shape optimization
        V = fd.FunctionSpace(mesh_m, "CG", 1)
        self.u0 = fd.Function(V)
        perturbation = 0.25*sin(x*pi)*sin(y*pi)**2
        self.u0.interpolate(sin(pi*x*(1+perturbation))*sin(pi*y))

        # heat equation discretized with implicit Euler
        self.u = fd.Function(V)
        v = fd.TestFunction(V)
        self.u_old = fd.Function(V)  # solution at previous time
        self.bcs = fd.DirichletBC(V, 0, "on_boundary")
        self.dx = fd.dx(metadata={"quadrature_degree": 1})
        self.dt = 0.125
        self.F = lambda t, u, u_old: fd.inner((u-u_old)/self.dt, v)*self.dx \
            + fd.inner(fd.grad(u), fd.grad(v))*self.dx \
            - fd.inner(self.f(t+self.dt), v)*self.dx

        # define self.cb, which is always called after self.solvePDE
        self.File = fd.VTKFile("u0.pvd")
        self.cb = lambda: self.File.write(self.u0)

        # function space and variable to assess mesh quality
        Vdet = fd.FunctionSpace(self.Q.mesh_r, "DG", 0)
        self.detDT = fd.Function(Vdet)

    def solvePDE(self):
        """Solve the heat equation and evaluate the objective function."""
        self.J = 0
        t = 0
        self.u.assign(self.u0)
        self.J += fd.assemble(self.dt*(self.u - self.u_t(t))**2*self.dx)

        for ii in range(10):
            self.u_old.assign(self.u)
            fd.solve(self.F(t, self.u, self.u_old) == 0, self.u, bcs=self.bcs)
            t += self.dt
            self.J += fd.assemble(self.dt*(self.u - self.u_t(t))**2*self.dx)

    def objective_value(self):
        """Return the value of the objective function."""
        self.detDT.interpolate(fd.det(fd.grad(self.Q.T)))
        mesh_is_fine = min(self.detDT.vector()) > 0.01
        if mesh_is_fine:
            return self.J
        else:
            from pyadjoint.adjfloat import AdjFloat
            import numpy as np
            return AdjFloat(np.NAN)


if __name__ == "__main__":

    # setup problem
    mesh = fd.UnitSquareMesh(20, 20)
    Q = fs.FeControlSpace(mesh)
    IP = fs.LaplaceInnerProduct(Q, fixed_bids=[1, 2, 3, 4])
    Q.assign_inner_product(IP)
    J = TimeTracking(Q)

    # PETSc.TAO solver using the limited-memory
    # variable-metric method. Call using
    # python timetracking.py -tao_monitor
    # to print updates in the terminal
    solver = PETSc.TAO().create()
    solver.setType("lmvm")
    solver.setFromOptions()
    solver.setSolution(Q.get_PETSc_zero_vec())
    solver.setObjectiveGradient(J.objectiveGradient, None)
    solver.setTolerances(gatol=1.0e-4, grtol=1.0e-4)
    solver.solve()
