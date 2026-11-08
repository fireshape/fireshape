"""Regression tests for MoYoSpectralConstraint.hessVec.

Run from the Fireshape checkout using the Firedrake environment:
    python -m pytest -q tests/test_moyo_hessian.py

Requires the patched fireshape/zoo/spectral_constraint.py in that checkout.
"""
import numpy as np
import pytest
import firedrake as fd
import fireshape as fs
from fireshape.zoo.spectral_constraint import (
    MoYoSpectralConstraint, _soft_threshold_jacobian,
)


def _svt(B, bound):
    U, s, Vt = np.linalg.svd(B, full_matrices=False)
    return U @ np.diag(np.maximum(s - bound, 0)) @ Vt


@pytest.mark.parametrize("dim", [2, 3])
@pytest.mark.parametrize("case", ["mixed", "all_active", "inactive",
                                  "repeated_active", "repeated_inactive",
                                  "zero", "zero_bound", "at_kink"])
def test_pointwise_svt_differential(dim, case):
    rng = np.random.default_rng(1234 + dim)
    U, _ = np.linalg.qr(rng.standard_normal((dim, dim)))
    V, _ = np.linalg.qr(rng.standard_normal((dim, dim)))
    bound = 0.5
    if case == "mixed":
        s = np.linspace(.25, .9, dim)
    elif case == "all_active":
        s = np.linspace(.8, 1.5, dim)
    elif case == "inactive":
        s = np.linspace(.1, .3, dim)
    elif case == "repeated_active":
        s = np.full(dim, 1.0)
    elif case == "repeated_inactive":
        s = np.full(dim, .2)
    elif case == "zero":
        s = np.zeros(dim)
    elif case == "zero_bound":
        s = np.linspace(0, 1, dim)
        bound = 0.0
    elif case == "at_kink":
        s = np.full(dim, bound)

    B = U @ np.diag(s) @ V.T
    E = rng.standard_normal((dim, dim))
    h = 1e-5
    exact = _soft_threshold_jacobian(B, E, bound)
    centred = (_svt(B + h*E, bound) - _svt(B - h*E, bound))/(2*h)

    # At the kink, compare with the symmetric *directional* difference:
    # no classical Hessian exists, but the midpoint selection matches
    # the central difference in the limit.
    rtol = 1e-3 if case == "at_kink" else 1e-6
    np.testing.assert_allclose(exact, centred, atol=rtol, rtol=rtol)


def _make_problem(bound, weight=2e3):
    mesh = fd.UnitSquareMesh(8, 8)
    Q = fs.FeControlSpace(mesh, add_to_degree_r=1, degree_c=2)
    inner = fs.H1InnerProduct(Q)
    q = fs.ControlVector(Q, inner)
    v = fs.ControlVector(Q, inner)
    w = fs.ControlVector(Q, inner)
    X, Y = fd.SpatialCoordinate(q.fun.function_space().mesh())
    bump = fd.sin(np.pi*X)**2 * fd.sin(np.pi*Y)**2
    cross = fd.sin(2*np.pi*X) * fd.sin(np.pi*Y)**2

    q.fun.interpolate(fd.as_vector((.085*bump, .032*bump)))
    v.fun.interpolate(fd.as_vector((bump + .2*cross, -.55*bump + .1*cross)))
    w.fun.interpolate(fd.as_vector((-.3*bump + .15*cross, bump + .2*cross)))
    J = MoYoSpectralConstraint(weight, fd.Constant(bound), Q)
    J.update(q, True, 0)
    return J, q, v, w


def _perturb(q, direction, alpha):
    q_new = q.clone()
    q_new.set(q)
    q_new.axpy(alpha, direction)
    return q_new


def _centred_gradient_action(J, q, v, h):
    g_plus = q.clone()
    g_minus = q.clone()
    J.gradient(g_plus, _perturb(q, v, +h), 1e-10)
    J.gradient(g_minus, _perturb(q, v, -h), 1e-10)
    result = q.clone()
    result.set(g_plus)
    result.axpy(-1, g_minus)
    result.scale(1/(2*h))
    J.update(q, True, -1)
    return result


def test_mwe_inactive_constraint_produces_zero_hessian():
    # MWE's default bound 0.5 is above all singular values of grad(S).
    J, q, v, _ = _make_problem(bound=.5)
    hv = q.clone()
    J.hessVec(hv, v, q, 1e-10)
    assert hv.norm() < 1e-9


def test_active_hessian_agrees_with_central_gradient_difference():
    J, q, v, _ = _make_problem(bound=.07)
    J.update_state()
    sigma = np.linalg.svd(J.lam_c_grad_S.dat.data_ro,
                          compute_uv=False)
    assert np.any(sigma > .07) and np.any(sigma < .07), \
        "Test must exercise both active and inactive singular values"

    hv = q.clone()
    J.hessVec(hv, v, q, 1e-10)
    assert hv.norm() > 1e-4
    for h in (1e-4, 5e-5):
        fd_hv = _centred_gradient_action(J, q, v, h)
        diff = q.clone()
        diff.set(fd_hv)
        diff.axpy(-1.0, hv)
        assert diff.norm()/hv.norm() < 2e-3


def test_nonzero_multiplier_hessian_matches_central_difference():
    J, q, v, _ = _make_problem(bound=.07)
    # Hold lambda fixed while differentiating the augmented Lagrangian.
    # A nonsymmetric multiplier exercises the left/right SVD factors.
    shift = np.array([[.012, .016], [-.008, .02]])
    J.lam.dat.data_wo[:] = J.c * shift
    J.update_state()

    hv = q.clone()
    J.hessVec(hv, v, q, 1e-10)
    fd_hv = _centred_gradient_action(J, q, v, 5e-5)
    diff = q.clone()
    diff.set(fd_hv)
    diff.axpy(-1.0, hv)
    assert diff.norm()/max(1.0, hv.norm()) < 2e-3


def test_hessian_is_symmetric():
    J, q, v, w = _make_problem(bound=.07)
    hv = q.clone()
    hw = q.clone()
    J.hessVec(hv, v, q, 1e-10)
    J.hessVec(hw, w, q, 1e-10)
    lhs, rhs = hv.dot(w), hw.dot(v)
    assert abs(lhs - rhs) <= 1e-7 * max(1., abs(lhs), abs(rhs))


def test_second_order_taylor_expansion():
    J, q, v, _ = _make_problem(bound=.07)
    gradient = q.clone()
    hv = q.clone()
    J.gradient(gradient, q, 1e-10)
    J.hessVec(hv, v, q, 1e-10)
    J.update(q, True, -1)
    J0 = float(J.value(q, 1e-10))
    Hvv = hv.dot(v)
    first_derivative = gradient.dot(v)
    assert Hvv > 0
    errors = []
    for h in (4e-3, 2e-3, 1e-3):
        qp, qm = _perturb(q, v, h), _perturb(q, v, -h)
        J.update(qp, True, -1)
        Jp = float(J.value(qp, 1e-10))
        J.update(qm, True, -1)
        Jm = float(J.value(qm, 1e-10))
        # Check the value/gradient contract independently of the Hessian.
        centred_slope = (Jp - Jm) / (2*h)
        assert abs(centred_slope - first_derivative) < 2e-3 * max(
            1., abs(first_derivative))
        # Symmetric value difference removes the first derivative.
        errors.append(abs((Jp + Jm - 2*J0) - h*h*Hvv))
    assert errors[-1] < 5e-3 * (1e-3**2) * Hvv
    # Check improvement on step halving without asserting an exact rate:
    # spatial projection and finite-precision roundoff may affect it.
    assert errors[-1] < errors[0]
