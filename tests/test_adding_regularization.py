import pytest
import firedrake as fd
import fireshape as fs
import fireshape.zoo as fsz

import ROL

def test_taylor_tests():
    n = 5
    mesh = fd.UnitSquareMesh(n, n)
    inner = fs.LaplaceInnerProduct()
    Q = fs.FeMultiGridControlSpace(mesh, inner, refinements=1, order=1)
    mesh_m = Q.mesh_m

    q = fs.ControlVector(Q)

    X = fd.SpatialCoordinate(mesh)
    q.fun.interpolate(0.5 * X)


    lower_bound = Q.T.copy(deepcopy=True)
    lower_bound.interpolate(fd.Constant((-0.0, -0.0)))
    upper_bound = Q.T.copy(deepcopy=True)
    upper_bound.interpolate(fd.Constant((+1.3, +0.9)))

    J1 = fsz.MoYoBoxConstraint(1, [1, 2, 3, 4], Q,
                               lower_bound=lower_bound,
                               upper_bound=upper_bound)
    J2 = fsz.MoYoSpectralConstraint(1, fd.Constant(0.2), Q)
    J3 = fsz.DeformationRegularization(Q, l2_reg=.1, sym_grad_reg=1.,
                                       skew_grad_reg=.5)
    J4 = fsz.CoarseDeformationRegularization(Q, l2_reg=.1, sym_grad_reg=1.,
                                             skew_grad_reg=.5)
    

    Js = 0.1 * J1 + J2 + 2. * (J3 + J4)
    g = q.clone()

    def run_taylor_test(J):
        J.update(q, None, 1)
        J.gradient(g, q, None)
        return J.checkGradient(q, g, 8, 1)

    def check_result(test_result):
        for i in range(len(test_result)-1):
            assert test_result[i+1][3] <= test_result[i][3] * 0.11

    check_result(run_taylor_test(J1))
    check_result(run_taylor_test(J2))
    check_result(run_taylor_test(J3))
    check_result(run_taylor_test(J4))
    check_result(run_taylor_test(Js))
