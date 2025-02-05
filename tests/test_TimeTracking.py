import pytest
import firedrake as fd
import fireshape as fs
from fireshape import PDEconstrainedObjective
import ROL
from pyadjoint.tape import get_working_tape, pause_annotation, annotate_tape


@pytest.fixture(autouse=True)
def handle_taping():
    yield
    tape = get_working_tape()
    tape.clear_tape()


@pytest.fixture(autouse=True, scope="function")
def handle_exit_annotation():
    yield
    # Since importing firedrake.adjoint modifies a global variable, we need to
    # pause annotations at the end of the module
    annotate = annotate_tape()
    if annotate:
        pause_annotation()


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
        x, y = fd.SpatialCoordinate(mesh_m)
        sin = fd.sin
        cos = fd.cos
        pi = fd.pi
        self.u_t = lambda t: sin(pi*x)*sin(pi*y)*cos(t)
        self.f = lambda t: sin(pi*x)*sin(pi*y)*(2*pi**2*cos(t) - sin(t))

        # perturped initial guess to be fixed by shape optimization
        V = fd.FunctionSpace(mesh_m, "CG", 1)
        self.u0 = fd.Function(V)
        perturbation = 0.25*sin(x*pi)*sin(y*pi)**2
        self.u0.interpolate(sin(pi*x*(1+perturbation))*sin(pi*y))

        # define self.cb, which is always called after self.solvePDE
        # self.VTKFile = fd.VTKFile("u0.pvd")
        # self.cb = lambda : self.VTKFile.write(self.u0)

        # heat equation discreized with implicit Euler
        self.u = fd.Function(V)
        self.u_old = fd.Function(V)  # solution at previous time
        self.bcs = fd.DirichletBC(V, 0, "on_boundary")
        self.dx = fd.dx(metadata={"quadrature_degree": 1})
        self.dt = 0.125
        v = fd.TestFunction(V)
        self.F = lambda t, u, u_old: fd.inner((u-u_old)/self.dt, v)*self.dx \
            + fd.inner(fd.grad(u), fd.grad(v))*self.dx \
            - self.f(t+self.dt)*v*self.dx

    def objective_value(self):
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
        return self.J


def test_TimeTracking():
    """ Main test."""

    # setup problem
    mesh = fd.UnitSquareMesh(20, 20)
    Q = fs.FeControlSpace(mesh)
    inner = fs.LaplaceInnerProduct(Q, fixed_bids=[1, 2, 3, 4])
    q = fs.ControlVector(Q, inner)

    # create PDEconstrained objective functional
    J = TimeTracking(Q)

    # ROL parameters
    params_dict = {
        'General': {'Secant': {'Type': 'Limited-Memory BFGS',
                               'Maximum Storage': 25}},
        'Step': {'Type': 'Trust Region'},
        'Status Test': {'Gradient Tolerance': 1e-3,
                        'Step Tolerance': 1e-8,
                        'Iteration Limit': 20}}

    # assemble and solve ROL optimization problem
    params = ROL.ParameterList(params_dict, "Parameters")
    problem = ROL.OptimizationProblem(J, q)
    solver = ROL.OptimizationSolver(problem, params)
    solver.solve()

    # verify that the norm of the gradient at optimum is small enough
    state = solver.getAlgorithmState()
    assert (state.gnorm < 1e-3)


if __name__ == '__main__':
    pytest.main()
