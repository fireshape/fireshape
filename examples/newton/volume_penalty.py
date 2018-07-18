import fireshape as fs
import firedrake as fd
from firedrake import Constant, grad, tr, div, dx


class DomainVolumePenalty(fs.ShapeObjective):

    """Volume constraint."""

    def __init__(self, Q, target_volume=None):
        super().__init__(Q)
        self.m_def_space = Q.V_m
        self.mesh = self.m_def_space.mesh()
        if target_volume is not None:
            self.target_volume = target_volume
        else:
            self.target_volume = self.current_volume()

    def current_volume(self):
        return fd.assemble(1 * fd.dx(domain=self.mesh))

    def non_squared_value(self):
        return self.current_volume() - self.target_volume

    def value(self, x, tol):
        return self.non_squared_value()**2

    def non_squared_derivative_form(self, wfun):
        return div(wfun) * dx

    def derivative_form(self, wfun):
        return 2 * Constant(self.non_squared_value()) * div(wfun) * dx

    def non_squared_second_derivative_exp(self, V, W):
        res = (div(V) * div(W) - tr(grad(V)*grad(W))) * dx
        return res

    def second_derivative_form(self, V, W):
        term1 = 2 * fd.assemble(self.non_squared_derivative_form(V))
        term1 *= self.non_squared_derivative_form(W)
        term2 = self.non_squared_second_derivative_exp(V, W) * 2 * self.non_squared_value()
        term1 += term2
        return term1

    def hessVec(self, hv, v, x, tol):
        Tv = fd.Function(self.V_r)
        v.controlspace.interpolate(v, Tv)
        Tvm = fd.Function(self.V_m, val=Tv)
        test = fd.TestFunction(self.V_m)

        term1 = fd.assemble(self.non_squared_derivative_form(test))
        term1 *= 2 * fd.assemble(self.non_squared_derivative_form(Tvm))
        term2 = fd.assemble(self.non_squared_second_derivative_exp(Tvm, test))
        term2 *= 2 * self.non_squared_value()
        term1 += term2
        self.deriv_m.assign(term1)

        v.controlspace.restrict(self.deriv_m, hv)
        hv.apply_riesz_map()
        hv.scale(self.scale)
