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


class L2tracking(PDEconstrainedObjective):
    """A Poisson BVP with hom DirBC as PDE constraint."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.mesh_m = self.Q.mesh_m

        # Setup problem
        self.V = fd.FunctionSpace(self.mesh_m, "CG", 1)

        # Preallocate solution variables for state and adjoint equations
        self.solution = fd.Function(self.V, name="State")

        # Weak form of Poisson problem
        u = self.solution
        v = fd.TestFunction(self.V)
        self.f = fd.Constant(4.)
        self.F = (fd.inner(fd.grad(u), fd.grad(v)) - self.f * v) * fd.dx
        self.bcs = fd.DirichletBC(self.V, 0., "on_boundary")

        # PDE-solver parameters
        self.params = {
            "ksp_type": "cg",
            "mat_type": "aij",
            "pc_type": "hypre",
            "pc_factor_mat_solver_package": "boomerang",
            "ksp_rtol": 1e-11,
            "ksp_atol": 1e-11,
            "ksp_stol": 1e-15,
        }

        stateproblem = fd.NonlinearVariationalProblem(
            self.F, self.solution, bcs=self.bcs)
        self.solver = fd.NonlinearVariationalSolver(
            stateproblem, solver_parameters=self.params)

        # target function, exact soln is disc of radius 0.6 centered at
        # (0.5,0.5)
        (x, y) = fd.SpatialCoordinate(self.mesh_m)
        self.u_target = 0.36 - (x-0.5)*(x-0.5) - (y-0.5)*(y-0.5)

    def objective_value(self):
        """Evaluate misfit functional. Signature imposed by ROL."""
        u = self.solution
        return fd.assemble((u - self.u_target)**2 * fd.dx)

    def solvePDE(self):
        self.solver.solve()


def run_L2tracking_optimization(write_output=False):
    """ Test template for fsz.LevelsetFunctional."""

    # tool for developing new tests, allows storing shape iterates
    if write_output:
        out = fd.File("domain.pvd")

        def cb(*args):
            out.write(Q.mesh_m.coordinates)

        cb()
    else:
        cb = None

    # setup problem
    mesh = fd.UnitSquareMesh(30, 30)
    Q = fs.FeControlSpace(mesh)
    inner = fs.ElasticityInnerProduct(Q)
    q = fs.ControlVector(Q, inner)

    # create PDEconstrained objective functional
    J = L2tracking(Q, cb=cb)

    # ROL parameters
    params_dict = {
        'General': {
            'Secant': {
                'Type': 'Limited-Memory BFGS',
                'Maximum Storage': 10
            }
        },
        'Step': {
            'Type': 'Line Search',
            'Line Search': {
                'Descent Method': {
                    'Type': 'Quasi-Newton Step'
                }
            },
        },
        'Status Test': {
            'Gradient Tolerance': 1e-4,
            'Step Tolerance': 1e-5,
            'Iteration Limit': 15
        }
    }

    # assemble and solve ROL optimization problem
    params = ROL.ParameterList(params_dict, "Parameters")
    problem = ROL.OptimizationProblem(J, q)
    solver = ROL.OptimizationSolver(problem, params)
    solver.solve()

    # verify that the norm of the gradient at optimum is small enough
    state = solver.getAlgorithmState()
    assert (state.gnorm < 1e-4)


def test_L2tracking(pytestconfig):
    verbose = False
    run_L2tracking_optimization(write_output=verbose)


if __name__ == '__main__':
    pytest.main()
