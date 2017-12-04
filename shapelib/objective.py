import ROL
import firedrake as fd

class Objective(ROL.Objective):

    def __init__(self, q):
        mesh = q.domain()
        self.mesh = mesh
        element = mesh.coordinates.function_space().ufl_element()
        self.V = fd.FunctionSpace(mesh, element)

    def value(self):
        raise NotImplementedError

    def value(self, x, tol):
        return self.value()

    def directional_derivative(self, v):
        raise NotImplementedError

    def gradient(self, g, x, tol):
        v = TestFunction(self.V)
        dir_deriv_vals = self.directional_derivative(v)
        q.inner_product.riesz_map(dir_deriv_vals, g)

    def update(self, x, flag, iteration):
        q.set(x)
        q.update_domain()
