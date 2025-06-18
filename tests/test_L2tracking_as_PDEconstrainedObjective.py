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
        f = fd.Constant(4.)
        F = (fd.inner(fd.grad(u), fd.grad(v)) - f * v) * fd.dx
        bcs = fd.DirichletBC(self.V, 0., "on_boundary")

        params = kwargs['solverparams']
        stateproblem = fd.NonlinearVariationalProblem(
            F, self.solution, bcs=bcs)
        self.solver = fd.NonlinearVariationalSolver(
            stateproblem, solver_parameters=params)

        # target function, exact soln is disc of radius 0.6 centered at
        # (0.5,0.5)
        (x, y) = fd.SpatialCoordinate(self.mesh_m)
        self.u_target = 0.36 - (x-0.5)*(x-0.5) - (y-0.5)*(y-0.5)

    def objective_value(self):
        """Evaluate objective function solving PDE constrained first."""
        self.solver.solve()
        u = self.solution
        return fd.assemble((u - self.u_target)**2 * fd.dx)


def run_L2tracking_optimization(controlspace, write_output=False):
    """ Test template for fsz.LevelsetFunctional."""

    # setup problem
    if controlspace == fs.FeControlSpace:
        mesh = fd.UnitSquareMesh(30, 30)
        Q = fs.FeControlSpace(mesh)
        pms = {
            "ksp_type": "cg",
            "mat_type": "aij",
            "pc_type": "hypre",
            "pc_factor_mat_solver_package": "boomerang",
            "ksp_rtol": 1e-11,
            "ksp_atol": 1e-11,
            "ksp_stol": 1e-15,
        }
    elif controlspace == fs.FeMultiGridControlSpace:
        mesh = fd.UnitSquareMesh(5, 5, diagonal="crossed")
        nref = 2
        mh = fd.MeshHierarchy(mesh, nref)
        Q = fs.FeMultiGridControlSpace(mh, coarse_control=True)
        # PDE-solver parameters
        pms = {"mat_type": "aij",
               "ksp_type": "cg",
               "ksp_rtol": 1.0e-14,
               "pc_type": "mg",
               "mg_coarse_pc_type": "cholesky",
               "mg_levels": {"ksp_max_it": 1,
                             "ksp_type": "chebyshev",
                             "pc_type": "jacobi"
                             }
               }
    inner = fs.H1InnerProduct(Q)
    q = fs.ControlVector(Q, inner)

    # tool for developing new tests, allows storing shape iterates
    if write_output:
        if controlspace == fs.FeControlSpace:
            out = fd.VTKFile("domain.pvd")

            def cb(*args):
                out.write(Q.mesh_m.coordinates)
        elif controlspace == fs.FeMultiGridControlSpace:
            names = ["D"+str(ii)+".pvd" for ii in range(nref+1)]
            out = [fd.VTKFile(name) for name in names]

            def cb(*args):
                for ii, D in enumerate(Q.mh_mapped):
                    out[ii].write(D.coordinates)
        cb()
    else:
        cb = None

    # create PDEconstrained objective functional
    print(pms)
    J = L2tracking(Q, cb=cb, solverparams=pms)

    # ROL parameters
    params_dict = {
        'General': {'Secant': {'Type': 'Limited-Memory BFGS',
                               'Maximum Storage': 10}},
        'Step': {'Type': 'Line Search',
                 'Line Search': {'Descent Method': {
                     'Type': 'Quasi-Newton Step'}}, },
        'Status Test': {'Gradient Tolerance': 1e-4,
                        'Step Tolerance': 1e-5,
                        'Iteration Limit': 15}
    }

    # assemble and solve ROL optimization problem
    params = ROL.ParameterList(params_dict, "Parameters")
    problem = ROL.OptimizationProblem(J, q)
    solver = ROL.OptimizationSolver(problem, params)
    solver.solve()

    # verify that the norm of the gradient at optimum is small enough
    state = solver.getAlgorithmState()
    assert (state.gnorm < 1e-4)

    # test that all multigrid meshes have been updated
    if controlspace == fs.FeMultiGridControlSpace:
        vol = [fd.assemble(fd.Constant(1)*fd.dx(D)) for D in Q.mh_mapped]
        import numpy as np
        assert np.allclose(vol, vol[0])


@pytest.mark.parametrize("controlspace", [fs.FeControlSpace,
                                          fs.FeMultiGridControlSpace])
def test_L2tracking(controlspace, pytestconfig):
    verbose = False
    run_L2tracking_optimization(controlspace, write_output=verbose)


if __name__ == '__main__':
    pytest.main()
