import firedrake as fd
from .control import ControlSpace


class Objective:
    """
    Abstract Objective class.

    Inputs:
    Q: ControlSpace
    cb: method to store current shape iterate at self.udpate
    quadrature_degree: quadrature degree to use. If None, then
    ufl will guesstimate the degree
    """
    def __init__(self, Q: ControlSpace, cb=None,
                 quadrature_degree: int = None):
        self.Q = Q  # ControlSpace
        self.V_r = Q.V_r  # fd.VectorFunctionSpace on reference mesh
        self.V_r_dual = Q.V_r_dual
        self.V_m = Q.V_m  # clone of V_r of physical mesh
        self.V_m_dual = Q.V_m_dual
        self.mesh_m = self.V_m.mesh()  # physical mesh
        self.cb = cb
        self.deriv_r = fd.Cofunction(self.V_r_dual)
        # container to store one gradient
        self.g_pre = Q.get_PETSc_zero_vec()
        if quadrature_degree is not None:
            self.params = {"quadrature_degree": quadrature_degree}
        else:
            self.params = None

    def value(self):
        """Evaluate objective functional."""
        raise NotImplementedError

    def derivative(self):
        """
        Compute the derivative of the objective (element in dual of space)
        and store it in self.deriv_r. Called by self.gradient.
        """
        raise NotImplementedError

    def gradient(self, g):
        """
        Compute Riesz representative of shape directional derivative.
        """
        self.derivative()  # store derivative in self.deriv_r
        self.Q.compute_gradient(self.deriv_r, g)

    def objectiveGradient(self, tao, x, g):
        """
        Evaluate objective and compute gradient. Signature imposed
        by TAO.setObjectiveGradient
        """
        if self.Q.update_domain(x):
            if self.cb is not None:
                self.cb()
            J = self.value()
            self.gradient(g)
            self.Jpre = J
            g.copy(self.g_pre)
            return J
        else:
            self.g_pre.copy(g)
            return self.Jpre

    def __add__(self, other):
        if isinstance(other, Objective):
            return ObjectiveSum(self, other)

    def __mul__(self, alpha):
        return ScaledObjective(self, alpha)

    def __rmul__(self, alpha):
        return ScaledObjective(self, alpha)


class UnconstrainedObjective(Objective):
    """Abstract class of unconstrained objective functionals."""
    def __init__(self, *args, **kwargs):
        """
        """
        super().__init__(*args, **kwargs)

    def value_form(self):
        """UFL formula of objective functional."""
        raise NotImplementedError

    def value(self):
        """Evaluate objective functional."""
        return fd.assemble(self.value_form(),
                           form_compiler_parameters=self.params)

    def derivative_form(self, v):
        """
        UFL formula of partial shape directional derivative
        """
        X = fd.SpatialCoordinate(self.mesh_m)
        return fd.derivative(self.value_form(), X, v)


class ShapeObjective(UnconstrainedObjective):
    """Abstract class of shape functionals."""
    def __init__(self, *args, **kwargs):
        """
        Construct a shape functional.

        Preallocate vectors for directional derivatives with respect to
        perturbations in self.V_m so that they are not created every time
        the derivative is evaluated (the same is done for the counterpart
        in self.V_r_dual in Objective.__init__).
        """
        super().__init__(*args, **kwargs)
        self.deriv_m = fd.Cofunction(self.V_m_dual)

    def derivative(self):
        """
        Assemble partial directional derivative wrt ControlSpace perturbations.

        First, assemble directional derivative (wrt FEspace V_m) and
        store it in self.deriv_m. Then pass the values to self.deriv_r.
        """
        v = fd.TestFunction(self.V_m)
        fd.assemble(self.derivative_form(v), tensor=self.deriv_m,
                    form_compiler_parameters=self.params)
        with self.deriv_m.dat.vec as vec_m:
            with self.deriv_r.dat.vec as vec_r:
                vec_m.copy(vec_r)


class DeformationObjective(UnconstrainedObjective):
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
        v = fd.TestFunction(self.V_r)
        fd.assemble(self.derivative_form(v), tensor=self.deriv_r,
                    form_compiler_parameters=self.params)

# comment out functionality which may be removed
# class ControlObjective(UnconstrainedObjective):
#    """
#    Similar to DeformationObjective, but in the case of a
#    FeMultigridConstrolSpace might want to formulate functionals
#    in term of the deformation defined on the coarse grid,
#    and not in terms of the prolonged deformation.
#    """
#
#    def __init__(self, *args, **kwargs):
#        super().__init__(*args, **kwargs)
#        # function used to define objective value_form,
#        # it contains the current control value, see self.update
#        self.f = fd.Function(self.Q.V_r_coarse)
#        # container for directional derivatives
#        self.deriv_r_coarse = fd.Cofunction(self.Q.V_r_coarse_dual)
#
#    def derivative(self, out):
#        """
#        Assemble partial directional derivative wrt ControlSpace
#        perturbations.
#        """
#        v = fd.TestFunction(self.Q.V_r_coarse)
#        fd.assemble(self.derivative_form(v), tensor=self.deriv_r_coarse,
#                    form_compiler_parameters=self.params)
#        out.cofun.assign(self.deriv_r_coarse)
#        out.scale(self.scale)


class PDEconstrainedObjective(Objective):
    """
    Abstract class of reduced PDE-constrained functionals.
    Shape differentiate using pyadjoint.
    """
    def __init__(self, *args, **kwargs):

        super().__init__(*args, **kwargs)
        # dummy variable pyadjoint uses to compute shape
        # derivative and evaluate shape functional. In practice,
        # self.T_m = 0 throughout the optimization because mesh_m
        # is update via Control.update_domain (because we keep mesh_r
        # fixed for the computation of shape gradients and BFGS)
        self.dT_m = fd.Function(self.Q.V_m)
        self.dT_r = fd.Function(self.Q.V_r)
        self.Jred = None

    def solvePDE(self):
        """
        User's implementation to solve the PDE constraint. This method is used
        by self.createJred to create the reduced functional.
        """
        raise NotImplementedError

    def objective_value(self):
        """
        User's implementation to evaluate the objective function once the PDE
        constraint has been solved. This method is used by self.createJred to
        create the reduced functional.
        """
        raise NotImplementedError

    def value(self):
        """
        Evaluate reduced objective.
        """
        if self.Jred is None:
            self.createJred()
        self.dT_r.assign(self.Q.T - self.Q.id)
        with self.dT_r.dat.vec_ro as a:
            with self.dT_m.dat.vec_wo as b:
                a.copy(b)
        return self.Jred.__call__(self.dT_m)

    def derivative(self):
        """
        Get the derivative from pyadjoint.
        """
        # use pyadjiont functionality to return cofunction
        opts = {"riesz_representation": None}
        dJ = self.Jred.derivative(options=opts)
        # copy coeffs of cofuntion on mesh_m into
        # coeffs of cofunction on mesh_r
        with dJ.dat.vec as vec_dJ:
            with self.deriv_r.dat.vec as vec_r:
                vec_dJ.copy(vec_r)

    def createJred(self):
        """Create reduced functional using pyadjiont."""
        try:
            # We use pyadjoint to calculate adjoint and shape derivatives,
            # in order to do this we need to "record a tape of the forward
            # solve", pyadjoint will then figure out all necessary
            # adjoints.
            import firedrake.adjoint as fda
            with fda.stop_annotating():
                mytape = fda.Tape()
                with fda.set_working_tape(mytape):
                    fda.continue_annotation()
                    mesh_m = self.Q.mesh_m
                    mesh_m.coordinates.assign(mesh_m.coordinates + self.dT_m)
                    self.solvePDE()
                    Jpyadj = self.objective_value()
                    self.Jred = fda.ReducedFunctional(Jpyadj,
                                                      fda.Control(self.dT_m))
        except fd.ConvergenceError:
            print("Failed to solve the state equation for initial guess.")
            raise fd.ConvergenceError


class ObjectiveSum(Objective):

    def __init__(self, a, b):
        super().__init__(a.Q)
        self.a = a
        self.b = b

    def value(self):
        return self.a.value() + self.b.value()

    def derivative(self):
        self.a.derivative()
        self.b.derivative()
        self.deriv_r = self.a.deriv_r + self.b.deriv_r


class ScaledObjective(Objective):

    def __init__(self, J, alpha):
        super().__init__(J.Q)
        self.J = J
        self.alpha = alpha

    def value(self):
        return self.alpha * self.J.value()

    def derivative(self, out):
        self.J.derivative()
        self.deriv_r *= self.alpha
