import pytest
import firedrake as fd
import fireshape as fs
import fireshape.zoo as fsz


@pytest.mark.parametrize("controlspace_t", [fs.FeControlSpace,
                                            fs.FeMultiGridControlSpace])
# @pytest.mark.parametrize("use_extension", [False]) , True])
@pytest.mark.parametrize("use_extension", [False])
def test_regularization(controlspace_t, use_extension):
    n = 10
    mesh = fd.UnitSquareMesh(n, n)

    if controlspace_t == fs.FeMultiGridControlSpace:
        mh = fd.MeshHierarchy(mesh, 1)
        Q = fs.FeMultiGridControlSpace(mh, coarse_control=True)
    else:
        Q = controlspace_t(mesh)

    if use_extension:
        inner = fs.SurfaceInnerProduct(Q)
        ext = fs.ElasticityExtension(Q.get_space_for_inner()[0])
    else:
        inner = fs.LaplaceInnerProduct(Q)
        ext = None

    q = fs.ControlVector(Q, inner, boundary_extension=ext)

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
    if isinstance(Q, fs.FeMultiGridControlSpace):
        J4 = fsz.CoarseDeformationRegularization(Q, l2_reg=.1, sym_grad_reg=1.,
                                                 skew_grad_reg=.5)
        Js = 0.1 * J1 + J2 + 2. * (J3+J4)
    else:
        Js = 0.1 * J1 + J2 + 2. * J3

    g = q.clone()

    def run_taylor_test(J):
        J.update(q, None, 1)
        J.gradient(g, q, None)
        return J.checkGradient(q, g, 7, 1)

    def check_result(test_result):
        for i in range(len(test_result)-1):
            assert test_result[i+1][3] <= test_result[i][3] * 0.11

    check_result(run_taylor_test(J1))
    check_result(run_taylor_test(J2))
    check_result(run_taylor_test(J3))
    if isinstance(Q, fs.FeMultiGridControlSpace):
        check_result(run_taylor_test(J4))
    check_result(run_taylor_test(Js))

@pytest.mark.parametrize("control_type, coarse_control", [
    ("fe", None),
    ("multigrid", True),
    ("multigrid", False),
])
def test_deformation_objective_hessian(control_type, coarse_control):
    mesh = fd.UnitSquareMesh(4, 4)

    if control_type == "fe":
        Q = fs.FeControlSpace(mesh)
    else:
        mh = fd.MeshHierarchy(mesh, 1)
        Q = fs.FeMultiGridControlSpace(mh, coarse_control=coarse_control)

    inner = fs.H1InnerProduct(Q, direct_solve=True)

    J = fsz.DeformationRegularization(Q)
    q = fs.ControlVector(Q, inner)
    v = q.clone()

    mesh_q = Q.mesh_r if control_type == "fe" else mh[0]
    x, y = fd.SpatialCoordinate(mesh_q)

    q.fun.interpolate(fd.as_vector((0.1 * x * y, -0.05 * x)))
    v.fun.interpolate(fd.as_vector((x * (1 - x), 0.3 * y * (1 - y))))

    Hv = q.clone()
    gp = q.clone()
    gm = q.clone()

    J.update(q, None, -1)
    J.hessVec(Hv, v, q, None)

    eps = 1e-5
    qp = q.clone()
    qm = q.clone()
    qp.set(q)
    qm.set(q)
    qp.axpy(eps, v)
    qm.axpy(-eps, v)

    J.update(qp, None, -1)
    J.gradient(gp, qp, None)

    J.update(qm, None, -1)
    J.gradient(gm, qm, None)

    gp.axpy(-1.0, gm)
    gp.scale(0.5 / eps)
    gp.axpy(-1.0, Hv)

    assert gp.norm() / Hv.norm() < 1e-6

def test_control_objective_hessian():
    mesh = fd.UnitSquareMesh(4, 4)
    mh = fd.MeshHierarchy(mesh, 1)
    Q = fs.FeMultiGridControlSpace(mh, coarse_control=True)
    inner = fs.H1InnerProduct(Q, direct_solve=True)

    J = fsz.CoarseDeformationRegularization(Q)
    q = fs.ControlVector(Q, inner)
    v = q.clone()

    x, y = fd.SpatialCoordinate(mesh)
    q.fun.interpolate(fd.as_vector((0.1 * x * y, -0.05 * x)))
    v.fun.interpolate(fd.as_vector((x * (1 - x), 0.3 * y * (1 - y))))

    Hv = q.clone()
    gp = q.clone()
    gm = q.clone()

    J.update(q, None, -1)
    J.hessVec(Hv, v, q, None)

    eps = 1e-5
    qp = q.clone()
    qm = q.clone()
    qp.set(q)
    qm.set(q)
    qp.axpy(eps, v)
    qm.axpy(-eps, v)

    J.update(qp, None, -1)
    J.gradient(gp, qp, None)

    J.update(qm, None, -1)
    J.gradient(gm, qm, None)

    gp.axpy(-1.0, gm)
    gp.scale(0.5 / eps)
    gp.axpy(-1.0, Hv)

    assert gp.norm() / Hv.norm() < 1e-6
