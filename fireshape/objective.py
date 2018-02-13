# import ROL
import _ROL as ROL
import firedrake as fd
from .control import ControlSpace, ControlVector


class Objective(ROL.Objective):

    def __init__(self, Q: ControlSpace, cb=None, scale=1.0):
        super().__init__()
        self.V_m = Q.V_m
        self.V_r = Q.V_r
        self.Q = Q
        self.cb = cb
        self.scale = scale

        """
        Create vectors for derivative in FE space and
        control space so that they are not created every time
        the derivative is evaluated.
        """

        self.deriv_m = fd.Function(self.V_m)
        self.deriv_r = fd.Function(self.V_r, val=self.deriv_m)
        self.deriv_control = ControlVector(Q)

    def value_form(self):
        raise NotImplementedError

    def value(self, x, tol):
        return self.scale * fd.assemble(self.value_form())

    def derivative_form(self, v):
        raise NotImplementedError

    def derivative(self):
        v = fd.TestFunction(self.V_m)
        fd.assemble(self.derivative_form(v), tensor=self.deriv_m)
        self.Q.restrict(self.deriv_r, self.deriv_control)
        self.deriv_control.scale(self.scale)
        return self.deriv_control

    def gradient(self, g, x, tol):
        dir_deriv_control = self.derivative()
        self.Q.inner_product.riesz_map(dir_deriv_control, g)

    def update(self, x, flag, iteration):
        self.Q.update_domain(x)
        if iteration > 0 and self.cb is not None:
            self.cb()
