import pytest
import firedrake as fd
import fireshape as fs
import fireshape.zoo as fsz
import ROL


@pytest.mark.parametrize("dim",
                         [2, 3])
@pytest.mark.parametrize("inner_t", [fs.H1InnerProduct,
                                     fs.ElasticityInnerProduct,
                                     fs.LaplaceInnerProduct])
@pytest.mark.parametrize("controlspace_t", [fs.FeControlSpace,
                                            fs.FeMultiGridControlSpace,
                                            fs.BsplineControlSpace])
@pytest.mark.parametrize("use_extension", ["wo_ext", "w_ext",
                                           "w_ext_fixed_fim"])
def test_levelset(dim, inner_t, controlspace_t, use_extension, pytestconfig):
    verbose = pytestconfig.getoption("verbose")
    """ Test template for fsz.LevelsetFunctional."""

    clscale = 0.1 if dim == 2 else 0.2

    # make the mesh a bit coarser if we are using a multigrid control space as
    # we are refining anyway
    if controlspace_t == fs.FeMultiGridControlSpace:
        clscale *= 2

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
        Q = fs.FeMultiGridControlSpace(mesh, refinements=1, order=2)
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

    if use_extension == "w_ext":
        ext = fs.ElasticityExtension(Q.V_r)
    if use_extension == "w_ext_fixed_dim":
        ext = fs.ElasticityExtension(Q.V_r, fixed_dims=[0])
    else:
        ext = None

    q = fs.ControlVector(Q, inner, boundary_extension=ext)

    """
    move mesh a bit to check that we are not doing the
    taylor test in T=id
    """
    g = q.clone()
    J.gradient(g, q, None)
    q.plus(g)
    J.update(q, None, 1)

    """ Start taylor test """
    J.gradient(g, q, None)
    res = J.checkGradient(q, g, 5, 1)
    errors = [l[-1] for l in res]
    assert (errors[-1] < 0.11 * errors[-2])
    q.scale(0)
    """ End taylor test """

    grad_tol = 1e-6 if dim == 2 else 1e-4
    # ROL parameters
    params_dict = {
        'General': {
            'Secant': {
                'Type': 'Limited-Memory BFGS',
                'Maximum Storage': 50
            }
        },
        'Step': {
            'Type': 'Line Search',
            'Line Search': {
                'Descent Method': {
                    'Type': 'Quasi-Newton Step'
                }
            }
        },
        'Status Test': {
            'Gradient Tolerance': grad_tol,
            'Step Tolerance': 1e-10,
            'Iteration Limit': 150
        }
    }

    # assemble and solve ROL optimization problem
    params = ROL.ParameterList(params_dict, "Parameters")
    problem = ROL.OptimizationProblem(J, q)
    solver = ROL.OptimizationSolver(problem, params)
    solver.solve()

    # verify that the norm of the gradient at optimum is small enough
    state = solver.getAlgorithmState()
    assert (state.gnorm < grad_tol)


if __name__ == '__main__':
    pytest.main()
