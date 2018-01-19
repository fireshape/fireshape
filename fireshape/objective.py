# import ROL
import _ROL as ROL
import firedrake as fd

class Objective(ROL.Objective):

    def __init__(self, q, cb=None):
        super().__init__()
        # self.V_m = q.V()
        # Om = q.controlspace.moving_mesh()
        self.V_m = q.controlspace.V_m_fine
        self.q = q
        self.Q = q.controlspace
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
        self.Q.update_mesh(x)
        # self.q.set(x)
        # self.q.update_domain()
        if iteration > 0 and self.cb is not None:
            self.cb()
