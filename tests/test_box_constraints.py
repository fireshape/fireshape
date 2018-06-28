import unittest
import firedrake as fd
import fireshape as fs
import fireshape.zoo as fsz
import numpy as np

import ROL


def test_box_constraint(pytestconfig):

    n = 5
    mesh = fd.UnitSquareMesh(n, n)
    T = mesh.coordinates.copy(deepcopy=True)
    (x, y) = fd.SpatialCoordinate(mesh)
    T.interpolate(T + fd.Constant((1, 0)) * x * y)
    mesh = fd.Mesh(T)

    Q = fs.FeControlSpace(mesh)
    inner = fs.LaplaceInnerProduct(Q, fixed_bids=[1])
    mesh_m = Q.mesh_m
    q = fs.ControlVector(Q, inner)
    if pytestconfig.getoption("verbose"):
        out = fd.File("domain.pvd")

        def cb(): out.write(mesh_m.coordinates)
    else:
        def cb(): pass

    lower_bound = Q.T.copy(deepcopy=True)
    lower_bound.interpolate(fd.Constant((-0.0, -0.0)))
    upper_bound = Q.T.copy(deepcopy=True)
    upper_bound.interpolate(fd.Constant((+1.3, +0.9)))

    J = fsz.MoYoBoxConstraint(1, [2], Q, lower_bound=lower_bound,
                              upper_bound=upper_bound,
                              cb=cb,
                              quadrature_degree=100)
    g = q.clone()
    J.gradient(g, q, None)
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
    Tvec = Q.T.vector()
    nodes = fd.DirichletBC(Q.V_r, fd.Constant((0.0, 0.0)), [2]).nodes
    assert np.all(Tvec[nodes, 0] <= 1.3 + 1e-4)
    assert np.all(Tvec[nodes, 1] <= 0.9 + 1e-4)


def test_objective_plus_box_constraint(pytestconfig):

    n = 10
    mesh = fd.UnitSquareMesh(n, n)
    T = mesh.coordinates.copy(deepcopy=True)
    (x, y) = fd.SpatialCoordinate(mesh)
    T.interpolate(T + fd.Constant((0, 0)))
    mesh = fd.Mesh(T)

    Q = fs.FeControlSpace(mesh)
    inner = fs.LaplaceInnerProduct(Q)
    mesh_m = Q.mesh_m
    q = fs.ControlVector(Q, inner)
    if pytestconfig.getoption("verbose"):
        out = fd.File("domain.pvd")

        def cb(): out.write(mesh_m.coordinates)
    else:
        def cb(): pass

    lower_bound = Q.T.copy(deepcopy=True)
    lower_bound.interpolate(fd.Constant((-0.2, -0.2)))
    upper_bound = Q.T.copy(deepcopy=True)
    upper_bound.interpolate(fd.Constant((+1.2, +1.2)))

    # levelset test case
    (x, y) = fd.SpatialCoordinate(Q.mesh_m)
    f = (pow(x-0.5, 2))+pow(y-0.5, 2) - 4.
    J1 = fsz.LevelsetFunctional(f, Q, cb=cb, quadrature_degree=10)
    J2 = fsz.MoYoBoxConstraint(10., [1, 2, 3, 4], Q, lower_bound=lower_bound,
                               upper_bound=upper_bound,
                               cb=cb,
                               quadrature_degree=10)
    J3 = fsz.MoYoSpectralConstraint(100, fd.Constant(0.6), Q, cb=cb,
                                    quadrature_degree=100)

    J = 0.1 * J1 + J2 + J3
    g = q.clone()
    J.gradient(g, q, None)
    taylor_result = J.checkGradient(q, g, 9, 1)

    for i in range(len(taylor_result)-1):
        if taylor_result[i][3] > 1e-6 and taylor_result[i][3] < 1e-3:
            assert taylor_result[i+1][3] <= taylor_result[i][3] * 0.15

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
            'Iteration Limit': 10}}

    params = ROL.ParameterList(params_dict, "Parameters")
    problem = ROL.OptimizationProblem(J, q)
    solver = ROL.OptimizationSolver(problem, params)
    solver.solve()
    Tvec = Q.T.vector()
    nodes = fd.DirichletBC(Q.V_r, fd.Constant((0.0, 0.0)), [2]).nodes
    assert np.all(Tvec[nodes, 0] <= 1.2 + 1e-1)
    assert np.all(Tvec[nodes, 1] <= 1.2 + 1e-1)


if __name__ == '__main__':
    unittest.main()
