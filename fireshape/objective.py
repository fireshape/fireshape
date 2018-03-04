import ROL
import firedrake as fd
from .control import ControlSpace, ControlVector
from .pde_constraint import PdeConstraint


class Objective(ROL.Objective):
    """Abstract class of shape functionals."""
    def __init__(self, Q: ControlSpace, cb=None, scale=1.0):
        """
        Construct a shape functional.

        Inputs: Q: type ControlSpace
                cb: method to store current shape iterate at self.udpate
                scale: type double, scaling factor that multiplies shape
                       functional and directional derivative

        Preallocate vectors for directional derivatives with respect to
        perturbations in self.V_m, for their clone on self.V_r, and for
        the directional derivative wrt perturbations in ControlSpace (so
        that they are not created every time the derivative is evaluated).
        Note that self.deriv_r is updated whenever self.deriv_m is.
        """
        super().__init__()
        self.Q = Q # ControlSpace
        self.V_r = Q.V_r # fd.VectorFunctionSpace on reference mesh
        self.V_m = Q.V_m # clone of V_r of physical mesh
        self.cb = cb
        self.scale = scale

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
        """
        Assemble directional derivative wrt ControlSpace perturbations.

        First, assemble directional derivative (wrt FEspace V_m) and
        store it in self.deriv_m. This automatically updates self.deriv_r,
        which is then converted to the directional derivative wrt
        ControSpace perturbations restrict.
        """
        v = fd.TestFunction(self.V_m)
        fd.assemble(self.derivative_form(v), tensor=self.deriv_m)
        self.Q.restrict(self.deriv_r, self.deriv_control)
        self.deriv_control.scale(self.scale)
        return self.deriv_control

    def gradient(self, g, x, tol):
        """Compute Riesz representative of directional derivative."""
        dir_deriv_control = self.derivative()
        self.Q.inner_product.riesz_map(dir_deriv_control, g)

    def update(self, x, flag, iteration):
        """Update physical domain and possibly store current iterate."""
        self.Q.update_domain(x)
        if iteration > 0 and self.cb is not None:
            self.cb()


class ReducedObjective(Objective):
    """Abstract class of reduces shape functionals."""
    def __init__(self, J: Objective, e: PdeConstraint):
        super().__init__(J.Q, J.cb)
        self.J = J
        self.e = e

    def value(self, x, tol):
        return self.J.value(x, tol)

    def derivative_form(self, v):
        return self.J.scale * self.J.derivative_form(v) + self.e.derivative_form(v)

    def update(self, x, flag, iteration):
        """Update domain and solution to state and adjoint equation."""
        self.Q.update_domain(x)
        self.e.solve()
        self.e.solve_adjoint(self.J.scale * self.J.value_form())
        if iteration > 0 and self.cb is not None:
            self.cb()
