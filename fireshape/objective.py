import ROL
import firedrake as fd
import firedrake_adjoint as fda
from .control import ControlSpace
from .pde_constraint import PdeConstraint
from .errors import MeshDeformationError
from mpi4py import MPI


class Objective(ROL.Objective):

    def __init__(self, Q: ControlSpace, cb=None,
                 quadrature_degree: int = None):

        """
        Inputs: Q: ControlSpace
                cb: method to store current shape iterate at self.udpate
                quadrature_degree: quadrature degree to use. If None, then
                ufl will guesstimate the degree
        """
        super().__init__()
        self.Q = Q  # ControlSpace
        self.V_r = Q.V_r  # fd.VectorFunctionSpace on reference mesh
        self.V_m = Q.V_m  # clone of V_r of physical mesh
        self.mesh_m = self.V_m.mesh()  # physical mesh
        self.cb = cb
        self.deriv_r = fd.Function(self.V_r)
        if quadrature_degree is not None:
            self.params = {"quadrature_degree": quadrature_degree}
        else:
            self.params = None

    def value(self, x, tol):
        """
        Value of the objective.
        Function signature imposed by ROL.
        """
        if not hasattr(self, "value_form"):
            raise NotImplementedError(
                "If you don't provide obj.value(x, tol), you need to "
                "provide a function `obj.value_form()`")
        return fd.assemble(self.value_form(),
                           form_compiler_parameters=self.params)

    def derivative(self, out):
        """
        Derivative of the objective (element in dual of space)
        """
        raise NotImplementedError

    def gradient(self, g, x, tol):
        """
        Compute Riesz representative of shape directional derivative.
        Function signature imposed by ROL.
        """

        self.derivative(g)
        g.apply_riesz_map()

    def update(self, x, flag, iteration):
        """Update physical domain and possibly store current iterate."""
        updated = self.Q.update_domain(x)
        if iteration >= 0 and self.cb is not None:
            self.cb()
        return updated

    def __add__(self, other):
        if isinstance(other, Objective):
            return ObjectiveSum(self, other)

    def __mul__(self, alpha):
        return ScaledObjective(self, alpha)

    def __rmul__(self, alpha):
        return ScaledObjective(self, alpha)


class ShapeObjective(Objective):
    """Abstract class of shape functionals."""
    def __init__(self, *args, **kwargs):
        """
        Construct a shape functional.

        Preallocate vectors for directional derivatives with respect to
        perturbations in self.V_m, for their clone on self.V_r, and for
        the directional derivative wrt perturbations in ControlSpace (so
        that they are not created every time the derivative is evaluated).
        Note that self.deriv_r is updated whenever self.deriv_m is.
        """
        super().__init__(*args, **kwargs)

        self.deriv_m = fd.Function(self.V_m, val=self.deriv_r)

    def derivative_form(self, v):
        if not hasattr(self, "value_form"):
            raise NotImplementedError(
                "If you don't provide obj.value_form(), you need to "
                "provide a function `obj.derivative_form(v)`")
        X = fd.SpatialCoordinate(self.mesh_m)
        return fd.derivative(self.value_form(), X, v)

    def derivative(self, out):
        """
        Assemble partial directional derivative wrt ControlSpace perturbations.

        First, assemble directional derivative (wrt FEspace V_m) and
        store it in self.deriv_m. This automatically updates self.deriv_r,
        which is then converted to the directional derivative wrt
        ControSpace perturbations restrict.
        """
        if not hasattr(self, "derivative_form"):
            raise NotImplementedError(
                "If you don't provide obj.derivative_form(v), you need to "
                "provide a function `obj.derivative(out)`")
        v = fd.TestFunction(self.V_m)
        fd.assemble(self.derivative_form(v), tensor=self.deriv_m,
                    form_compiler_parameters=self.params)
        out.from_first_derivative(self.deriv_r)


class DeformationObjective(Objective):
    """
    Abstract class for functionals that depend on the deformation of the mesh.
    These are different from shape functionals, as they are entirely defined on
    the reference mesh. Examples are regularizing functionals like
    J(f) = int |nabla(f)| dx.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def derivative(self, out):
        """
        Assemble partial directional derivative wrt ControlSpace perturbations.
        """
        if not hasattr(self, "derivative_form"):
            raise NotImplementedError(
                "If you don't provide obj.derivative_form(v), you need to "
                "provide a function `obj.derivative(out)`")
        v = fd.TestFunction(self.V_r)
        fd.assemble(self.derivative_form(v), tensor=self.deriv_r,
                    form_compiler_parameters=self.params)
        out.from_first_derivative(self.deriv_r)


class ControlObjective(Objective):

    """
    Similar to DeformationObjective, but in the case of a
    FeMultigridConstrolSpace might want to formulate functionals
    in term of the deformation defined on the coarse grid,
    and not in terms of the prolonged deformation.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        (self.V_control, I) = self.Q.get_space_for_inner()
        assert I is None
        self.deriv_r = fd.Function(self.V_control)

    def derivative(self, out):
        """
        Assemble partial directional derivative wrt ControlSpace perturbations.
        """
        if not hasattr(self, "derivative_form"):
            raise NotImplementedError(
                "If you don't provide obj.derivative_form(v), you need to "
                "provide a function `obj.derivative(out)`")
        v = fd.TestFunction(self.V_control)
        fd.assemble(self.derivative_form(v), tensor=self.deriv_r,
                    form_compiler_parameters=self.params)
        out.fun.assign(self.deriv_r)

    # def update(self, x, flag, iteration):
    #     self.f.assign(x.fun)
    #     super().update(x, flag, iteration)


class ReducedObjective(ShapeObjective):
    """Abstract class of reduced shape functionals."""
    def __init__(self, J: Objective, e: PdeConstraint):
        if not isinstance(J, ShapeObjective):
            msg = "PDE constraints are currently only supported" \
                + " for shape objectives."
            raise NotImplementedError(msg)

        super().__init__(J.Q, J.cb)
        self.J = J
        self.e = e
        # stop any annotation that might be ongoing as we only want to record
        # what's happening in e.solve()
        fda.pause_annotation()

    def value(self, x, tol):
        """
        Evaluate reduced objective.
        Function signature imposed by ROL.
        """
        return self.J.value(x, tol)

    def derivative(self, out):
        """
        Get the derivative from pyadjoint.
        """

        out.from_first_derivative(self.Jred.derivative())

    def update(self, x, flag, iteration):
        """Update domain and solution to state and adjoint equation."""
        if super().update(x, flag, iteration):
            try:
                # We use pyadjoint to calculate adjoint and shape derivatives,
                # in order to do this we need to "record a tape of the forward
                # solve", pyadjoint will then figure out all necessary
                # adjoints.
                tape = fda.get_working_tape()
                tape.clear_tape()
                fda.continue_annotation()
                mesh_m = self.J.Q.mesh_m
                s = fda.Function(self.J.V_m)
                mesh_m.coordinates.assign(mesh_m.coordinates + s)
                self.s = s
                self.c = fda.Control(s)
                self.e.solve()
                Jpyadj = fda.assemble(self.J.value_form(),
                                      form_compiler_parameters=self.params)
                self.Jred = fda.ReducedFunctional(Jpyadj, self.c)
                fda.pause_annotation()
            except fd.ConvergenceError:
                if self.cb is not None:
                    self.cb()
                raise


class ObjectiveSum(Objective):

    def __init__(self, a, b):
        super().__init__(a.Q)
        self.a = a
        self.b = b

    def value(self, x, tol):
        return self.a.value(x, tol) + self.b.value(x, tol)

    def value_form(self):
        return self.a.value_form() + self.b.value_form()

    def derivative(self, out):
        temp = out.clone()
        self.a.derivative(out)
        self.b.derivative(temp)
        out.plus(temp)

    def derivative_form(self, v):
        return self.a.derivative_form(v) + self.b.derivative_form(v)

    def update(self, x, flag, iteration):
        resa = self.a.update(x, flag, iteration)
        resb = self.b.update(x, flag, iteration)
        if iteration >= 0 and self.cb is not None:
            self.cb()
        return resa or resb


class ScaledObjective(Objective):

    def __init__(self, J, alpha):
        super().__init__(J.Q)
        self.J = J
        self.alpha = alpha

    def value(self, *args):
        return self.alpha * self.J.value(*args)

    def derivative(self, out):
        self.J.derivative(out)
        out.scale(self.alpha)

    def update(self, x, flag, iteration):
        res = self.J.update(x, flag, iteration)
        if iteration >= 0 and self.cb is not None:
            self.cb()
        return res


class DeformationCheckObjective(Objective):
    """

Wrapping this around an objective will prevent deformations that are larger
than a given threshold.  If strict=False, then we do not raise an exception,
but instead we return the same value as at the previous iteration and the same
gradient but with a flipped sign.  The idea behind that is to make the
optimization think that we have stepped over the minimizer (as in the picture
below) which makes it half its stepsize and try again.

^
|  X               X
|  XX             XX
|   XX           XX
|    XX         XX
|     XX       XX
|      XXX   XXX
|       XXXXXXX
|
+------------------->
  x_k          x_{k+1}

    """

    def __init__(self, J, threshold=None, delta_threshold=None, strict=True,
                 cb=None):
        super().__init__(J.Q)
        self.J = J
        self.threshold = threshold
        self.strict = strict
        self.delta_threshold = delta_threshold
        self.cb = cb
        self.last_value = None
        self.last_derivative = None
        self.return_last = False
        self.X = J.Q.T.copy(deepcopy=True)
        # self.gradX = fd.Function(fd.TensorFunctionSpace(
        #     self.X.ufl_domain(), "DG", self.X.ufl_element().degree()))
        self.gradX = fd.Function(fd.TensorFunctionSpace(
            self.X.ufl_domain(), "DG", 0))
        self.lastX = self.X.copy(deepcopy=True)
        self.lastX *= 0
        self.diff = self.lastX.copy(deepcopy=True)

    def value(self, *args):
        if not self.return_last:
            self.last_value = self.J.value(*args)
        if self.return_last is None:
            raise RuntimeError("This shouldn't happen.")
        return self.last_value

    def derivative(self, out):
        if self.return_last:
            out.set(self.last_derivative)
            out.scale(-1.)
        else:
            if self.last_derivative is None:
                self.last_derivative = out.clone()
            self.J.derivative(self.last_derivative)
            out.set(self.last_derivative)

    def check_singular_values(self, X, eps):
        """
        The singular values of the deformation are a good way of checking
        that it is not `too large`.  One can show that on convex domains,
        sigma(grad(X)) < 1 is a sufficient condition for T = Id + X being a
        diffeomorphism.
        """

        from numpy.linalg import svd
        self.gradX.project(fd.grad(X))
        gradXv = self.gradX.vector()
        # dim = X.ufl_shape[0]
        # max_val = 0
        for i, M in enumerate(gradXv):
            W, Sigma, V = svd(M, full_matrices=False)
            if Sigma[0] > eps:
                fd.warning(fd.RED % ("Deformation too large."))
                return False
            # for j in range(dim):
            #     # if Sigma[j] > max_val:
            #     #     max_val = Sigma[j]
            #     if Sigma[j] > eps:
            #         # print(Sigma[j])
            #         return False
        # print("Biggest value in check_singular_values", max_val)
        # return max_val < eps
        return True

    def update(self, q, *args):
        q.to_coordinatefield(self.X)
        comm = self.X.ufl_domain().comm
        if self.delta_threshold is not None:
            self.diff.assign(self.X-self.lastX)
            check = self.check_singular_values(self.diff, self.delta_threshold)
            check = comm.allreduce(check, MPI.LAND)
            if not check:
                self.return_last = True
                if self.cb is not None:
                    self.cb()
                if self.strict:
                    raise MeshDeformationError(
                        "Mesh deformation of this step too large.")
                return
            else:
                self.lastX.assign(self.X)
        if self.threshold is not None:
            check = self.check_singular_values(self.X, self.threshold)
            check = comm.allreduce(check, MPI.LAND)
            if not check:
                self.return_last = True
                if self.cb is not None:
                    self.cb()
                if self.strict:
                    raise MeshDeformationError(
                        "Total mesh deformation too large.")
                return
        self.return_last = False
        self.J.update(q, *args)
