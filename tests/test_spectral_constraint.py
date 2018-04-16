import unittest
import firedrake as fd
import fireshape as fs
import fireshape.zoo as fsz
import numpy as np

import ROL


def test_spectral_constraint(pytestconfig):
    n = 5
    mesh = fd.UnitSquareMesh(n, n)
    T = mesh.coordinates.copy(deepcopy=True)
    T.interpolate(T - fd.Constant((0.5, 0.5)))
    mesh = fd.Mesh(T)
    inner = fs.LaplaceInnerProduct()
    Q = fs.FeControlSpace(mesh, inner)
    mesh_m = Q.mesh_m
    q = fs.ControlVector(Q)
    if pytestconfig.getoption("verbose"):
        out = fd.File("domain.pvd")

        def cb(): out.write(mesh_m.coordinates)
    else:
        def cb(): pass

    J = fsz.MoYoSpectralConstraint(0.5, fd.Constant(0.1), Q,
                                   cb=cb)
    q.fun += Q.T
    g = q.clone()
    J.update(q, None, -1)
    J.gradient(g, q, None)
    cb()
    taylor_result = J.checkGradient(q, g, 9, 1)

    for i in range(len(taylor_result)-1):
        if taylor_result[i][3] > 1e-7:
            assert taylor_result[i+1][3] <= taylor_result[i][3] * 0.11

    params_dict = {
        'General': {
            'Secant': {'Type': 'Limited-Memory BFGS',
                       'Maximum Storage': 2}},
        'Step': {
            'Type': 'Line Search',
            'Line Search': {'Descent Method': {
                'Type': 'Quasi-Newton Step'}}},
        'Status Test': {
            'Gradient Tolerance': 1e-10,
            'Step Tolerance': 1e-10,
            'Iteration Limit': 150}}

    params = ROL.ParameterList(params_dict, "Parameters")
    problem = ROL.OptimizationProblem(J, q)
    solver = ROL.OptimizationSolver(problem, params)
    solver.solve()
    Tvec = Q.T.vector()[:, :]
    for i in range(Tvec.shape[0]):
        assert abs(Tvec[i, 0]) < 0.55 + 1e-4
        assert abs(Tvec[i, 1]) < 0.55 + 1e-4
    assert np.any(np.abs(Tvec) > 0.55 - 1e-4)


if __name__ == '__main__':
    unittest.main()
