import pytest
import firedrake as fd
import fireshape as fs
import fireshape.zoo as fsz
import ROL


@pytest.mark.parametrize("dim", [2, 3])
@pytest.mark.parametrize("add_to_degree_r", [0, 1])
@pytest.mark.parametrize("inner_t", [fs.H1InnerProduct,
                                     fs.ElasticityInnerProduct,
                                     fs.LaplaceInnerProduct])
@pytest.mark.parametrize("decoupled", [False, True])
def test_levelset(dim, add_to_degree_r, inner_t, decoupled, pytestconfig):
    verbose = pytestconfig.getoption("verbose")
    """ Test template for fsz.LevelsetFunctional."""

    if dim == 2:
        mesh_r = fd.UnitDiskMesh()
    elif dim == 3:
        mesh_r = fd.UnitBallMesh()
    else:
        raise NotImplementedError

    if decoupled:
        degree_c = 2
        if dim == 2:
            mesh_c = fd.RectangleMesh(10, 10, 2, 2, -2, -2)
        else:
            # 5 cells per direction, height=width=depth=6
            mesh_c = fd.BoxMesh(5, 5, 5, 6, 6, 6)
            # shift in x-, y-, and z-direction
            mesh_c.coordinates.dat.data[:, 0] -= 3
            mesh_c.coordinates.dat.data[:, 1] -= 3
            mesh_c.coordinates.dat.data[:, 2] -= 3
    else:
        mesh_c = None
        degree_c = None
    Q = fs.FeControlSpace(mesh_r, add_to_degree_r, mesh_c, degree_c)

    inner = inner_t(Q)
    # if running with -v or --verbose, then export the shapes
    if verbose:
        out = fd.VTKFile("domain.pvd")

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
    errors = [L[-1] for L in res]
    assert (errors[-1] < 0.11 * errors[-2])
    q.scale(0)
    """ End taylor test """

    grad_tol = 1e-5 if dim == 2 else 1e-4
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
