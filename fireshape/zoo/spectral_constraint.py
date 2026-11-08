import firedrake as fd
import fireshape as fs
import numpy as np
from numpy.linalg import svd

__all__ = ["MoYoSpectralConstraint"]


def _soft_threshold_jacobian(B, E, threshold):
    """Directional derivative of singular-value soft thresholding.

    For B = U diag(s) V^T and F(B) = U diag((s-threshold)_+) V^T,
    return DF(B)[E]. This is an ordinary derivative away from the
    nonsmooth set {singular value == threshold}. At the kink we choose
    the midpoint (1/2) of the active/inactive scalar derivatives, i.e.,
    a generalized Hessian element, rather than claiming differentiability.

    Both B and E must be square, real matrices (2x2 or 3x3 in Fireshape).
    """
    if threshold < 0:
        raise ValueError("Spectral constraint threshold must be nonnegative")
    # With zero threshold, soft thresholding is the identity map,
    # including at singular matrices. Avoid 0/0 in divided differences.
    if threshold == 0:
        return E.copy()

    U, s, Vt = svd(B, full_matrices=False)
    f = np.maximum(s - threshold, 0.0)
    X = U.T @ E @ Vt.T
    n = len(s)
    Y = np.empty_like(X)

    # Relative tolerance is only used to classify equality at the kink.
    tol = 16.0 * np.finfo(s.dtype).eps * max(1.0, threshold, s[0])
    for i in range(n):
        if s[i] > threshold + tol:
            dfi = 1.0
        elif s[i] < threshold - tol:
            dfi = 0.0
        else:
            dfi = 0.5  # generalized derivative on the kink
        Y[i, i] = dfi * X[i, i]

        for j in range(i + 1, n):
            # Symmetric divided difference of f(s)=(s-threshold)_+.
            # Branching by activity avoids subtractive cancellation for
            # equal or nearly equal singular values.
            if (abs(s[i] - threshold) <= tol and
                    abs(s[j] - threshold) <= tol):
                alpha = 0.5  # repeated singular values at the kink
            elif s[i] > threshold and s[j] > threshold:
                alpha = 1.0
            elif s[i] < threshold and s[j] < threshold:
                alpha = 0.0
            else:
                alpha = (f[i] - f[j]) / (s[i] - s[j])

            # Skew-symmetric divided difference; denominator can vanish
            # only if s[i]=s[j]=0, in which case f[i]=f[j]=0 for threshold>0.
            beta = (f[i] + f[j]) / (s[i] + s[j]) if s[i] + s[j] else 0.0
            sym = 0.5 * (X[i, j] + X[j, i])
            skew = 0.5 * (X[i, j] - X[j, i])
            Y[i, j] = alpha * sym + beta * skew
            Y[j, i] = alpha * sym - beta * skew

    return U @ Y @ Vt


class MoYoSpectralConstraint(fs.DeformationObjective):

    def __init__(self, c, bound, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.c = c
        self.T = self.Q.T
        r_mesh = self.T.ufl_domain()
        self.lam_space = fd.TensorFunctionSpace(r_mesh, "DG", 0)
        self.scalar_space = fd.FunctionSpace(r_mesh, "DG", 0)
        self.nuclear_norm = fd.Function(self.scalar_space)
        self.viol = fd.Function(self.scalar_space)
        self.lam = fd.Function(self.lam_space)
        self.gradS = fd.Function(self.lam_space)
        self.lam_c_grad_S = fd.Function(self.lam_space)
        self.argmin = fd.Function(self.lam_space)
        self.bound = fd.Function(self.scalar_space).interpolate(bound)

        self.iden = fd.Function(self.V_r)
        self.iden.interpolate(fd.SpatialCoordinate(r_mesh))
        self.S = self.T.copy(deepcopy=True)
        self.dim = r_mesh.topological_dimension
        # Storage for the projected direction and the derivative of argmin.
        self.hess_grad = fd.Function(self.lam_space)
        self.hess_dargmin = fd.Function(self.lam_space)

    def update_state(self):
        lam = self.lam
        c = self.c
        av = self.bound.dat.data_ro
        self.S.assign(self.T)
        self.S -= self.iden
        S = self.S
        self.gradS.project(fd.grad(S))
        lam_c_grad_S = self.lam_c_grad_S
        lam_c_grad_S.project(lam/c + self.gradS)
        lam_c_grad_Sv = lam_c_grad_S.dat.data_ro
        for i in range(len(lam_c_grad_Sv)):
            W, Sigma, V = svd(lam_c_grad_Sv[i], full_matrices=False)
            for j in range(self.dim):
                Sigma[j] = c * max(Sigma[j]-av[i], 0)
            self.argmin.dat.data_wo[i] = np.dot(W, np.dot(np.diag(Sigma), V))
            self.nuclear_norm.dat.data_wo[i] = av[i]*np.sum(Sigma)

    def value_form(self):
        self.update_state()
        val = fd.inner(self.gradS, self.argmin) * fd.dx \
            - self.nuclear_norm * fd.dx \
            - (0.5/self.c) * fd.inner(self.argmin - self.lam,
                                      self.argmin - self.lam) * fd.dx
        return val

    def derivative_form(self, test):
        self.update_state()
        return fd.inner(self.argmin, fd.grad(test))*fd.dx

    def hessVec(self, hv, v, x, tol):
        """Riesz representative of the Hessian action (multiplier fixed).

        ``argmin`` is computed via a NumPy SVD, so UFL differentiation of
        derivative_form() would miss its dependence on the deformation.
        At a threshold singular value, this uses a generalized Hessian.
        """
        if (v.boundary_extension is not None
                or hv.boundary_extension is not None):
            raise NotImplementedError(
                "Hessian actions with boundary_extension are not supported."
            )

        self.Q.update_domain(x)
        self.update_state()
        v.to_coordinatefield(self.hess_dir_r)

        # Differentiate the *same* DG0 projection as used in update_state.
        self.hess_grad.project(fd.grad(self.hess_dir_r))
        B = self.lam_c_grad_S.dat.data_ro
        E = self.hess_grad.dat.data_ro
        bounds = self.bound.dat.data_ro
        out = self.hess_dargmin.dat.data_wo
        for i in range(len(B)):
            out[i] = self.c * _soft_threshold_jacobian(B[i], E[i], bounds[i])

        w = fd.TestFunction(self.V_r)
        fd.assemble(fd.inner(self.hess_dargmin, fd.grad(w)) * fd.dx,
                    tensor=self.deriv_r,
                    form_compiler_parameters=self.params)
        hv.from_first_derivative(self.deriv_r)
        hv.scale(self.scale)
        hv.apply_riesz_map()

    def update_multiplier(self, stepsize=1.0):
        lam = self.lam
        lam *= (1-stepsize)
        lam += stepsize * self.argmin
        self.update_state()

    def violation(self):
        gradSv = self.gradS.dat.data_ro
        av = self.upper_bound.dat.data_ro
        for i in range(len(gradSv)):
            W, Sigma, V = svd(gradSv[i], full_matrices=False)
            for j in range(self.dim):
                Sigma[j] = max(Sigma[j]-av[i], 0)**2
            self.viol.dat.data_wo[i] = self.c * 0.5 * np.sum(Sigma)
        return fd.sqrt(fd.assemble(self.viol*fd.dx))
