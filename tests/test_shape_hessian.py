import pytest
import numpy as np
import firedrake as fd
import fireshape as fs


class GeometryObjective(fs.ShapeObjective):
    def value_form(self):
        x, y = fd.SpatialCoordinate(self.mesh_m)
        return (fd.sin(x) + x * y + 0.5 * y**3) * fd.dx


class VolumeObjective(fs.ShapeObjective):
    def value_form(self):
        return fd.Constant(1.0) * fd.dx(domain=self.mesh_m)

@pytest.mark.parametrize(
    "control_type, coarse_control",
    [
        ("fe", None),
        ("multigrid", True),
        ("multigrid", False),
    ],
    ids=["fe", "mg-coarse", "mg-fine"],
)
def test_shape_hessian_action(control_type, coarse_control):
    mesh = fd.UnitSquareMesh(4, 4)

    if control_type == "fe":
        Q = fs.FeControlSpace(mesh)
        mesh_q = mesh
    else:
        mh = fd.MeshHierarchy(mesh, 1)
        Q = fs.FeMultiGridControlSpace(mh, coarse_control=coarse_control)
        mesh_q = mh[0] if coarse_control else mh[-1]

    inner = fs.H1InnerProduct(Q, direct_solve=True)
    J = GeometryObjective(Q)

    q = fs.ControlVector(Q, inner)
    v = q.clone()
    w = q.clone()

    x, y = fd.SpatialCoordinate(mesh_q)

    q.fun.interpolate(fd.as_vector((0.03 * x, -0.02 * y)))

    v.fun.interpolate(
        fd.as_vector((0.10 * (1 + x) * y, -0.08 * x * (1 + y)))
    )

    w.fun.interpolate(
        fd.as_vector((-0.07 * x * (1 + y), 0.09 * (1 + x) * y))
    )

    J.update(q, None, -1)

    Hv = q.clone()
    Hw = q.clone()

    J.hessVec(Hv, v, q, None)
    J.hessVec(Hw, w, q, None)

    v_r = fd.Function(Q.V_r)
    v_m = fd.Function(Q.V_m)
    v.to_coordinatefield(v_r)

    with v_r.dat.vec_ro as vec_r:
        with v_m.dat.vec_wo as vec_m:
            vec_r.copy(vec_m)

    X = fd.SpatialCoordinate(Q.mesh_m)
    Hvv_form = fd.derivative(fd.derivative(J.value_form(), X, v_m), X, v_m)
    Hvv = fd.assemble(Hvv_form)

    assert np.isclose(v.dot(Hv), Hvv)
    assert np.isclose(w.dot(Hv), v.dot(Hw))

    eps = 1e-5

    qp = q.clone()
    qm = q.clone()
    qp.set(q)
    qm.set(q)
    qp.axpy(eps, v)
    qm.axpy(-eps, v)

    gp = q.clone()
    gm = q.clone()

    J.update(qp, None, -1)
    J.gradient(gp, qp, None)

    J.update(qm, None, -1)
    J.gradient(gm, qm, None)

    Hfd = q.clone()
    Hfd.set(gp)
    Hfd.axpy(-1.0, gm)
    Hfd.scale(0.5 / eps)

    error = q.clone()
    error.set(Hfd)
    error.axpy(-1.0, Hv)

    assert error.norm() / Hv.norm() < 1e-5

def test_shape_hessian_kernel():
    mesh = fd.UnitSquareMesh(4, 4)

    Q = fs.FeControlSpace(mesh)
    inner = fs.H1InnerProduct(Q, direct_solve=True)
    J = VolumeObjective(Q)

    q = fs.ControlVector(Q, inner)
    v = q.clone()
    Hv = q.clone()

    x, y = fd.SpatialCoordinate(mesh)
    bubble = x * (1 - x) * y * (1 - y)

    v.fun.interpolate(fd.as_vector((bubble, -bubble)))

    J.update(q, None, -1)
    J.hessVec(Hv, v, q, None)

    assert Hv.norm() < 1e-10

def test_objective_sum_hessian():
    mesh = fd.UnitSquareMesh(4, 4)
    Q = fs.FeControlSpace(mesh)
    inner = fs.H1InnerProduct(Q, direct_solve=True)

    J1 = GeometryObjective(Q)
    J2 = VolumeObjective(Q)
    J = J1 + J2

    q = fs.ControlVector(Q, inner)
    v = q.clone()

    x, y = fd.SpatialCoordinate(Q.mesh_r)
    v.fun.interpolate(fd.as_vector((0.2 * x * y, -0.1 * x * (1 + y))))

    H1 = q.clone()
    H2 = q.clone()
    H = q.clone()

    J.update(q, None, -1)
    J1.hessVec(H1, v, q, None)
    J2.hessVec(H2, v, q, None)
    J.hessVec(H, v, q, None)

    H1.plus(H2)
    H.axpy(-1.0, H1)

    assert H.norm() < 1e-10


def test_scaled_objective_hessian():
    mesh = fd.UnitSquareMesh(4, 4)
    Q = fs.FeControlSpace(mesh)
    inner = fs.H1InnerProduct(Q, direct_solve=True)

    J1 = GeometryObjective(Q)
    alpha = 2.7
    J = alpha * J1

    q = fs.ControlVector(Q, inner)
    v = q.clone()

    x, y = fd.SpatialCoordinate(Q.mesh_r)
    v.fun.interpolate(fd.as_vector((0.2 * x * y, -0.1 * x * (1 + y))))

    H1 = q.clone()
    H = q.clone()

    J.update(q, None, -1)
    J1.hessVec(H1, v, q, None)
    J.hessVec(H, v, q, None)

    H1.scale(alpha)
    H.axpy(-1.0, H1)

    assert H.norm() < 1e-10
