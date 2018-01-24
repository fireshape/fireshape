# import ROL
import _ROL as ROL
import firedrake as fd
from .control import ControlSpace

class Objective(ROL.Objective):

    def __init__(self, Q: ControlSpace, cb=None):
        super().__init__()
        self.V_m = Q.V_m
        self.Q = Q
        self.cb = cb

    def val(self):
        raise NotImplementedError

    def value(self, x, tol):
        return self.val()

    def derivative_form(self, v):
        raise NotImplementedError

    def derivative(self):
        v = fd.TestFunction(self.V_m)
        dir_deriv_fem = fd.assemble(self.derivative_form(v))
        dir_deriv_control = self.Q.restrict(dir_deriv_fem)
        return dir_deriv_control
        
    def gradient(self, g, x, tol):
        dir_deriv_control = self.derivative()
        self.Q.inner_product.riesz_map(dir_deriv_control, g)

    def update(self, x, flag, iteration):
        self.Q.update_domain(x)
        if iteration > 0 and self.cb is not None:
            self.cb()
