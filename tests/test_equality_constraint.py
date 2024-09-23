import unittest
import firedrake as fd
import fireshape as fs
import fireshape.zoo as fsz

import ROL


def test_equality_constraint(pytestconfig):
    mesh = fs.DiskMesh(0.05, radius=2.)

    Q = fs.FeControlSpace(mesh)
    inner = fs.ElasticityInnerProduct(Q, direct_solve=True)
    mesh_m = Q.mesh_m
    (x, y) = fd.SpatialCoordinate(mesh_m)

    q = fs.ControlVector(Q, inner)
    if pytestconfig.getoption("verbose"):
        out = fd.VTKFile("domain.pvd")

        def cb(*args):
            out.write(Q.mesh_m.coordinates)
    else:
        cb = None
    f = (pow(2*x, 2))+pow(y-0.1, 2) - 1.2

    J = fsz.LevelsetFunctional(f, Q, cb=cb)
    vol = fsz.LevelsetFunctional(fd.Constant(1.0), Q)
    e = fs.EqualityConstraint([vol])
    emul = ROL.StdVector(1)

    params_dict = {
        'Step': {
            'Type': 'Augmented Lagrangian',
            'Augmented Lagrangian': {
                'Subproblem Step Type': 'Line Search',
                'Penalty Parameter Growth Factor': 2.,
                'Initial Penalty Parameter': 1.,
                'Subproblem Iteration Limit': 20,
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
                'Maximum Storage': 5
            }
        },
        'Status Test': {
            'Gradient Tolerance': 1e-4,
            'Step Tolerance': 1e-10,
            'Iteration Limit': 10
        }
    }

    params = ROL.ParameterList(params_dict, "Parameters")
    problem = ROL.OptimizationProblem(J, q, econ=e, emul=emul)
    solver = ROL.OptimizationSolver(problem, params)
    solver.solve()

    state = solver.getAlgorithmState()
    assert (state.gnorm < 1e-4)
    assert (state.cnorm < 1e-6)


if __name__ == '__main__':
    unittest.main()
