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

    inner = fs.LaplaceInnerProduct(fixed_bids=[1])
    Q = fs.FeControlSpace(mesh, inner)
    mesh_m = Q.mesh_m
    out = fd.File("domain.pvd")
    q = fs.ControlVector(Q)
    if pytestconfig.getoption("verbose"):
        def cb(i): out.write(mesh_m.coordinates)
    else:
        def cb(i): pass

    lower_bound = Q.T.copy(deepcopy=True)
    lower_bound.interpolate(fd.Constant((-0.0, -0.0)))
    upper_bound = Q.T.copy(deepcopy=True)
    upper_bound.interpolate(fd.Constant((+1.3, +0.9)))

    J = fsz.MoYoBoxConstraint(1, [2], Q, lower_bound=lower_bound,
                              upper_bound=upper_bound,
                              cb=lambda: out.write(mesh_m.coordinates),
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


if __name__ == '__main__':
    unittest.main()
