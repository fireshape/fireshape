import firedrake as fd
from .control import ControlSpace
from .pde_constraint import PdeConstraint


class Objective:

    def __init__(self, Q: ControlSpace, cb=None, scale: float = 1.0,
                 quadrature_degree: int = None):

        """
        Inputs: Q: ControlSpace
                cb: method to store current shape iterate at self.udpate
                scale: scaling factor that multiplies shape
                       functional and directional derivative
                quadrature_degree: quadrature degree to use. If None, then
                ufl will guesstimate the degree
        """
        super().__init__()
        self.Q = Q  # ControlSpace
        self.V_r = Q.V_r  # fd.VectorFunctionSpace on reference mesh
        self.V_r_dual = Q.V_r_dual
        self.V_m = Q.V_m  # clone of V_r of physical mesh
        self.V_m_dual = Q.V_m_dual
        self.mesh_m = self.V_m.mesh()  # physical mesh
        self.cb = cb
        self.scale = scale
        self.deriv_r = fd.Cofunction(self.V_r_dual)
        if quadrature_degree is not None:
            self.params = {"quadrature_degree": quadrature_degree}
        else:
            self.params = None

    def value_form(self):
        """UFL formula of misfit functional."""
        raise NotImplementedError

    def value(self, x, tol):
        """Evaluate misfit functional. Function signature imposed by ROL."""
        return self.scale * fd.assemble(self.value_form(),
                                        form_compiler_parameters=self.params)

    def derivative_form(self, v):
        """
        UFL formula of partial shape directional derivative
        """
        X = fd.SpatialCoordinate(self.mesh_m)
        return fd.derivative(self.value_form(), X, v)

    def derivative(self, out):
        """
        Derivative of the objective (element in dual of space).
        Called by self.gradient.
        """
        raise NotImplementedError

    def gradient(self, g, x, tol):
        """
        Compute Riesz representative of shape directional derivative.
        Function signature imposed by ROL.
        """
        self.derivative(g)
        g.apply_riesz_map()

#    def update(self, x, flag, iteration):
#        """Update physical domain and possibly store current iterate."""
#        self.Q.update_domain(x)
#        if iteration >= 0 and self.cb is not None:
#            self.cb()
#
    def __add__(self, other):
        if isinstance(other, Objective):
            return ObjectiveSum(self, other)

    def __mul__(self, alpha):
        return ScaledObjective(self, alpha)

    def __rmul__(self, alpha):
        return ScaledObjective(self, alpha)

class UnconstrainedObjective(Objective)
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
        return self.scale * fd.assemble(self.value_form(),
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
        store it in self.deriv_m. Then pass the values to self.deriv_r,
        which is then converted to the directional derivative wrt
        ControlSpace perturbations using ControlSpace.restrict .
        """
        v = fd.TestFunction(self.V_m)
        fd.assemble(self.derivative_form(v), tensor=self.deriv_m,
                    form_compiler_parameters=self.params)
        with self.deriv_m.dat.vec as vec_m:
            with self.deriv_r.dat.vec as vec_r:
                vec_m.copy(vec_r)
        #self.Q.from_first_derivative(self.deriv_r)
        #out.scale(self.scale)


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
        #out.from_first_derivative(self.deriv_r)
        #out.scale(self.scale)


#class ControlObjective(UnconstrainedObjective):
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
#        Assemble partial directional derivative wrt ControlSpace perturbations.
#        """
#        v = fd.TestFunction(self.Q.V_r_coarse)
#        fd.assemble(self.derivative_form(v), tensor=self.deriv_r_coarse,
#                    form_compiler_parameters=self.params)
#        out.cofun.assign(self.deriv_r_coarse)
#        out.scale(self.scale)
#
#    #def update(self, x, flag, iteration):
#    #    self.f.assign(x.fun)
#    #    super().update(x, flag, iteration)


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
        self.T_m = fd.Function(self.Q.V_m)

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
        return self.Jred.__call__(self.T_m)

    def derivative(self):
        """
        Get the derivative from pyadjoint.
        """
        # use pyadjiont functionality to return cofunction
        opts = {"riesz_representation": None}
        dJ = self.Jred.derivative(options=opts)
        # copy coeffs of cofuntion on mesh_m into
        # coeffs of cofunction on mesh_r
        # Note: consider definining a method for this
        with dJ.dat.vec as vec_dJ:
            with self.deriv_r.dat.vec as vec_r:
                vec_dJ.copy(vec_r)
        #self.Q.from_first_derivative(self.deriv_r)

    def createJred(self):
        """Create reduced functional using pyadjiont."""
        try:
            # We use pyadjoint to calculate adjoint and shape derivatives,
            # in order to do this we need to "record a tape of the forward
            # solve", pyadjoint will then figure out all necessary
            # adjoints.
            import firedrake.adjoint as fda
            with stop_annotating():
                mytape = fda.Tape()
                with set_working_tape(mytape):
                    fda.continue_annotation()
                    mesh_m = self.Q.mesh_m
                    mesh_m.coordinates.assign(mesh_m.coordinates + self.T_m)
                    self.solvePDE()
                    Jpyadj = self.objective_value()
                    self.Jred = fda.ReducedFunctional(Jpyadj, fda.Control(self.T_m)
        except fd.ConvergenceError:
            print("Failed to solve the state equation for initial guess.")
            raise

#    def update(self, x, flag, iteration):
#        """Update domain and solution to state and adjoint equation."""
#        if self.Q.update_domain(x):
#            try:
#                # We use pyadjoint to calculate adjoint and shape derivatives,
#                # in order to do this we need to "record a tape of the forward
#                # solve", pyadjoint will then figure out all necessary
#                # adjoints.
#                import firedrake.adjoint as fda
#                fda.continue_annotation()
#                tape = fda.get_working_tape()
#                tape.clear_tape()
#                # ensure we are annotating
#                from pyadjoint.tape import annotate_tape
#                safety_counter = 0
#                while not annotate_tape():
#                    safety_counter += 1
#                    fda.continue_annotation()
#                    if safety_counter > 1e2:
#                        import sys
#                        sys.exit('Cannot annotate even after 100 attempts.')
#                mesh_m = self.Q.mesh_m
#                s = fd.Function(self.Q.V_m)
#                mesh_m.coordinates.assign(mesh_m.coordinates + s)
#                self.s = s
#                self.c = fda.Control(s)
#                self.solvePDE()
#                Jpyadj = self.objective_value()
#                self.Jred = fda.ReducedFunctional(Jpyadj, self.c)
#                fda.pause_annotation()
#            except fd.ConvergenceError:
#                if self.cb is not None:
#                    self.cb()
#                raise
#        if iteration >= 0 and self.cb is not None:
#            self.cb()


class ObjectiveSum(Objective):

    def __init__(self, a, b):
        super().__init__(a.Q)
        self.a = a
        self.b = b

    def value(self):
        return self.a.value(x, tol) + self.b.value(x, tol)

    def derivative(self, out):
        temp = out.clone()
        self.a.derivative(out)
        self.b.derivative(temp)
        out.plus(temp)

#    def update(self, *args):
#        self.a.update(*args)
#        self.b.update(*args)


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

#    def update(self, *args):
#        self.J.update(*args)
