import firedrake as fd
import fireshape as fs
import numpy as np
from numpy.linalg import svd

__all__ = ["MoYoSpectralConstraint"]


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
        # from firedrake import File
        # self.lam_file = File("lam.pvd")
        self.S = self.T.copy(deepcopy=True)
        self.dim = r_mesh.geometric_dimension()

    def update_state(self):
        lam = self.lam
        c = self.c
        av = self.bound.vector()[:]
        self.S.assign(self.T)
        self.S -= self.iden
        S = self.S
        self.gradS.project(fd.grad(S))
        lam_c_grad_S = self.lam_c_grad_S
        lam_c_grad_S.project(lam/c + self.gradS)
        lam_c_grad_Sv = lam_c_grad_S.vector()
        B = self.argmin.vector()[:].copy()
        nucv = self.nuclear_norm.vector()[:].copy()
        for i in range(len(lam_c_grad_Sv)):
            W, Sigma, V = svd(lam_c_grad_Sv[i], full_matrices=False)
            for j in range(self.dim):
                Sigma[j] = c * max(Sigma[j]-av[i], 0)
            B[i] = np.dot(W, np.dot(np.diag(Sigma), V))
            nucv[i] = av[i]*np.sum(Sigma)
        self.argmin.vector().set_local(B.flatten())
        self.nuclear_norm.vector().set_local(nucv.flatten())

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

    def update_multiplier(self, stepsize=1.0):
        lam = self.lam
        lam *= (1-stepsize)
        lam += stepsize * self.argmin
        self.update_state()

    def violation(self):
        gradSv = self.gradS.vector()[:]
        av = self.upper_bound.vector()[:]
        violv = self.viol.vector()[:].copy()
        for i in range(len(gradSv)):
            W, Sigma, V = svd(gradSv[i], full_matrices=False)
            for j in range(self.dim):
                Sigma[j] = max(Sigma[j]-av[i], 0)**2
            violv[i] = self.c * 0.5 * np.sum(Sigma)
        self.viol.vector().set_local(violv.flatten())
        return fd.sqrt(fd.assemble(self.viol*fd.dx))
