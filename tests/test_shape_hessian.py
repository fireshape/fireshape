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


def test_shape_hessian_action():
    mesh = fd.UnitSquareMesh(4, 4)

    Q = fs.FeControlSpace(mesh)
    inner = fs.H1InnerProduct(Q, direct_solve=True)

    J = GeometryObjective(Q)

    q = fs.ControlVector(Q, inner)
    v = q.clone()
    w = q.clone()

    x, y = fd.SpatialCoordinate(Q.mesh_r)

    # Do not test only at the identity deformation.

    q.fun.interpolate(fd.as_vector((0.03 * x, -0.02 * y,)))

    v.fun.interpolate(
        fd.as_vector((
            0.10 * (1 + x) * y,
            -0.08 * x * (1 + y),
        ))
    )

    w.fun.interpolate(
        fd.as_vector((
            -0.07 * x * (1 + y),
            0.09 * (1 + x) * y,
        ))
    )

    J.update(q, None, -1)

    Hv = q.clone()
    Hw = q.clone()

    J.hessVec(Hv, v, q, None)
    J.hessVec(Hw, w, q, None)

    # Compare the quadratic form with UFL directly
    v_r = fd.Function(Q.V_r)
    v_m = fd.Function(Q.V_m)

    v.to_coordinatefield(v_r)

    with v_r.dat.vec_ro as vec_r:
        with v_m.dat.vec_wo as vec_m:
            vec_r.copy(vec_m)

    X = fd.SpatialCoordinate(Q.mesh_m)

    Hvv_form = fd.derivative(fd.derivative(J.value_form(), X, v_m), X, v_m,)

    Hvv = fd.assemble(Hvv_form)

    # Check symmetry
    assert np.isclose(v.dot(Hv), Hvv, rtol=1e-10, atol=1e-12)

    # Compare against a centred finite difference of Fireshape gradients
    eps = 1e-5

    qp = q.clone()
    qm = q.clone()
    qp.set(q)
    qm.set(q)

    qp.axpy(+eps, v)
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
    Hfd.scale(1.0 / (2.0 * eps))

    error = q.clone()
    error.set(Hfd)
    error.axpy(-1.0, Hv)

    rel_error = error.norm() / Hv.norm()

    assert rel_error < 1e-5


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
