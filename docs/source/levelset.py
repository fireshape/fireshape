# Example 1: Level Set
# ====================
#
# In this example, we show how to minimize the shape function
#
# .. math::
#
#     \mathcal{J}(\Omega) = \int_\Omega f(\mathbf{x}) \,\mathrm{d}\mathbf{x}\,.
#
# where :math:`f:\mathbb{R}^2\to\mathbb{R}` is a scalar function.
# In particular, we consider
#
# .. math::
#
#     f(x,y) = (x - 0.5)^2 + (y - 0.5)^2 - 0.64\,.
#
# The domain that minimizes this shape functional is a
# disc of radius :math:`0.8` centered at :math:`(0.5,0.5)`.
#
#
# Import modules
# ^^^^^^^^^^^^^^
#
# We begin by importing Firedrake, Fireshape, and ROL. ::

from firedrake import *
from fireshape import *
import ROL

# Implement the shape function
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# To implement the shape function :math:`\mathcal{J}`, we use Fireshape's class
# :bash:`PDEconstrainedObjective`. This requires specifying how to evaluate
# :math:`\mathcal{J}` in the method
# :bash:`PDEconstrainedObjective.objective_value`. (Although :math:`\mathcal{J}`
# is not technically constrained to a boundary value problem, it is
# convenient to use the class  :bash:`PDEconstrainedObjective` as this
# automatically returns :bash:`NaN` on poor quality meshes.) ::

class LevelsetFunction(PDEconstrainedObjective):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # physical mesh
        mesh_m = self.Q.mesh_m

        # integrand defined in terms of physical coordinates
        x, y = SpatialCoordinate(mesh_m)
        self.f = (x - 0.5)**2 + (y - 0.5)**2 - 0.64

    def objective_value(self):
        return assemble(self.f * dx)

# Select initial guess, control space, and inner product
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# We select a unit disk centered at the origin as initial domain.
# To modify the domain, we create a control space of geometric
# transformations discretized using finite elements. To compute
# descent directions, we employ Riesz representatives of shape
# derivatives with respect to a full :math:`H^1`-inner product.
# With these, we create a control variable :bash:`q` that will
# be updated by the optimization algorithm. ::

mesh = UnitDiskMesh(refinement_level=3)
Q = FeControlSpace(mesh)
IP = H1InnerProduct(Q)
q = ControlVector(Q, IP)

# Instantiate objective function J
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# We instantiate :math:`J(\Omega)` using the class
# :bash:`LevelsetFunction` we have created. During instantiation,
# we also pass a call back function :bash:`cb` that stores the
# shape iterates whenever :bash:`J` is evaluated. ::

out = VTKFile("levelset_domain.pvd")
J = LevelsetFunction(Q, cb=lambda: out.write(Q.mesh_m.coordinates))

# Select the optimization algorithm and solve the problem
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# Finally, we select a trust-region optimization algorithm with l-BFGS Hessian
# updates and set the optimization stopping criteria in the dictionary
# :bash:`pd`.  This, together with :bash:`J` and :bash:`q` are passed to ROL,
# which solves the problem. ::

pd = {'Step': {'Type': 'Trust Region'},
      'General':  {'Secant': {'Type': 'Limited-Memory BFGS',
                                       'Maximum Storage': 25}},
       'Status Test': {'Gradient Tolerance': 1e-3,
                       'Step Tolerance': 1e-8,
                       'Iteration Limit': 30}}
params = ROL.ParameterList(pd, "Parameters")
problem = ROL.OptimizationProblem(J, q)
solver = ROL.OptimizationSolver(problem, params)
solver.solve()

# Result
# ^^^^^^
# Typing :bash:`python3 levelset.py` in the terminal returns:
#
# .. code-block:: none
#
#     Quasi-Newton Method with Limited-Memory BFGS
#     Line Search: Cubic Interpolation satisfying Strong Wolfe Conditions
#       iter  value          gnorm          snorm          #fval     #grad     ls_#fval  ls_#grad
#       0     -3.333333e-01  2.426719e-01
#       2     -3.886767e-01  8.040797e-02   2.525978e-01   3         3         1         0
#       3     -3.919733e-01  2.331695e-02   6.622577e-02   4         4         1         0
#       4     -3.926208e-01  4.331993e-03   5.628606e-02   5         5         1         0
#       5     -3.926603e-01  2.906313e-03   1.239420e-02   6         6         1         0
#       6     -3.926945e-01  9.456530e-04   2.100085e-02   7         7         1         0
#       7     -3.926980e-01  3.102278e-04   6.952015e-03   8         8         1         0
#       8     -3.926987e-01  1.778454e-04   3.840828e-03   9         9         1         0
#       9     -3.926989e-01  9.001788e-05   2.672387e-03   10        10        1         0
#     Optimization Terminated with Status: Converged
#
#
# We can inspect the result by opening the file :bash:`domain.pvd`
# with `ParaView <https://www.paraview.org/>`_.
# Example 1: Level Set
