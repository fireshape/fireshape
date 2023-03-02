import pytest
import firedrake as fd
import fireshape as fs
import fireshape.zoo as fsz
import ROL


@pytest.mark.parametrize("dim", [2, 3])
def test_wavelet_control_space(dim, pytestconfig):
    verbose = pytestconfig.getoption("verbose")
    """Test template for fs.WaveletControlSpace."""

    clscale = 0.1 if dim == 2 else 0.2

    if dim == 2:
        mesh = fs.DiskMesh(clscale)
        bbox = [(-2, 2), (-2, 2)]
        orders = [2, 2]
        levels = [4, 4]
    elif dim == 3:
        mesh = fs.SphereMesh(clscale)
        bbox = [(-3, 3), (-3, 3), (-3, 3)]
        orders = [2, 2, 2]
        levels = [3, 3, 3]
    else:
        raise NotImplementedError

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
            'Gradient Tolerance': 1e-6,
            'Step Tolerance': 1e-10,
            'Iteration Limit': 10
        }
    }
    params = ROL.ParameterList(params_dict, "Parameters")

    Q1 = fs.WaveletControlSpace(mesh, bbox, orders, orders, levels,
                                homogeneous_bc=[False]*dim)
    Q2 = fs.BsplineControlSpace(mesh, bbox, orders, levels,
                                boundary_regularities=[0]*dim)

    values = []
    gnorms = []
    for Q in [Q1, Q2]:
        inner = fs.H1InnerProduct(Q)
        # if running with -v or --verbose, then export the shapes
        if verbose and isinstance(Q, fs.WaveletControlSpace):
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

        # assemble and solve ROL optimization problem
        problem = ROL.OptimizationProblem(J, q)
        solver = ROL.OptimizationSolver(problem, params)
        solver.solve()

        state = solver.getAlgorithmState()
        values.append(state.value)
        gnorms.append(state.gnorm)

    # verify that J and gnorms after 10 iterations are same
    assert abs(values[0] - values[1]) < 1e-8
    assert abs(gnorms[0] - gnorms[1]) < 1e-8


if __name__ == '__main__':
    pytest.main()
