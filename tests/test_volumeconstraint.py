import pytest
import firedrake as fd
import fireshape as fs
import fireshape.zoo as fsz
import ROL


@pytest.mark.parametrize("inner_t", [fs.H1InnerProduct,
                                     fs.ElasticityInnerProduct,
                                     fs.LaplaceInnerProduct])
@pytest.mark.parametrize("controlspace_t", [fs.FeControlSpace,
                                            fs.FeMultiGridControlSpace,
                                            fs.BsplineControlSpace])
@pytest.mark.parametrize("dim", [2, 3])
def test_levelset(dim, inner_t, controlspace_t, pytestconfig):
    verbose = pytestconfig.getoption("verbose")
    """ Test template for fsz.LevelsetFunctional."""

    clscale = 0.1 if dim == 2 else 0.2

    # make the mesh a bit coarser if we are using a multigrid control space as
    # we are refining anyway
    if controlspace_t == fs.FeMultiGridControlSpace:
        clscale *= 4

    if dim == 2:
        mesh = fs.DiskMesh(clscale)
    elif dim == 3:
        mesh = fs.SphereMesh(clscale)
    else:
        raise NotImplementedError

    if controlspace_t == fs.BsplineControlSpace:
        if dim == 2:
            bbox = [(-2, 2), (-2, 2)]
            orders = [2, 2]
            levels = [4, 4]
        else:
            bbox = [(-3, 3), (-3, 3), (-3, 3)]
            orders = [2, 2, 2]
            levels = [3, 3, 3]
        Q = fs.BsplineControlSpace(mesh, bbox, orders, levels)
    elif controlspace_t == fs.FeMultiGridControlSpace:
        Q = fs.FeMultiGridControlSpace(mesh, refinements=1, degree=2)
    else:
        Q = controlspace_t(mesh)

    inner = inner_t(Q)
    # if running with -v or --verbose, then export the shapes
    if verbose:
        out = fd.File("domain.pvd")

        def cb(*args):
            out.write(Q.mesh_m.coordinates)

        cb()
    else:
        cb = None

    # levelset test case
    if dim == 2:
        (x, y) = fd.SpatialCoordinate(Q.mesh_m)
        f = (pow(x, 2))+pow(1.3*y, 2) - 1.
    elif dim == 3:
        (x, y, z) = fd.SpatialCoordinate(Q.mesh_m)
        f = (pow(x, 2))+pow(0.8*y, 2)+pow(1.3 * z, 2) - 1.

    else:
        raise NotImplementedError

    J = fsz.LevelsetFunctional(f, Q, cb=cb, scale=0.1)
    q = fs.ControlVector(Q, inner)

    # these tolerances are not very stringent, but solutions are correct with
    # tighter tolerances,  the combination
    # FeMultiGridControlSpace-ElasticityInnerProduct fails because the mesh
    # self-intersects (one should probably be more careful with the opt params)
    grad_tol = 1e-1
    itlim = 15
    itlimsub = 15

    # Volume constraint
    vol = fsz.LevelsetFunctional(fd.Constant(1.0), Q, scale=1)
    initial_vol = vol.value(q, None)
    econ = fs.EqualityConstraint([vol], target_value=[initial_vol])
    emul = ROL.StdVector(1)

    # ROL parameters
    params_dict = {
        'Step': {
            'Type': 'Augmented Lagrangian',
            'Augmented Lagrangian': {
                'Subproblem Step Type': 'Line Search',
                'Penalty Parameter Growth Factor': 1.05,
                'Print Intermediate Optimization History': True,
                'Subproblem Iteration Limit': itlimsub
            },
            'Line Search': {
                'Descent Method': {
                    'Type': 'Quasi-Newton Step'
                }
            },
        },
        'General': {
            'Secant': {
                'Type': 'Limited-Memory BFGS',
                'Maximum Storage': 50
            }
        },
        'Status Test': {
            'Gradient Tolerance': grad_tol,
            'Step Tolerance': 1e-10,
            'Iteration Limit': itlim
        }
    }
    params = ROL.ParameterList(params_dict, "Parameters")
    problem = ROL.OptimizationProblem(J, q, econ=econ, emul=emul)
    solver = ROL.OptimizationSolver(problem, params)
    solver.solve()

    # verify that the norm of the gradient at optimum is small enough
    # and that the volume has not changed too much
    state = solver.getAlgorithmState()
    assert (state.gnorm < grad_tol)
    assert abs(vol.value(q, None) - initial_vol) < 1e-2


if __name__ == '__main__':
    pytest.main()
