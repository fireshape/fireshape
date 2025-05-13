Example 1: Level Set
====================

In this example, we show how to minimize the shape function

.. math::

    \mathcal{J}(\Omega) = \int_\Omega f(\mathbf{x}) \,\mathrm{d}\mathbf{x}\,.

where :math:`f:\mathbb{R}^2\to\mathbb{R}` is a scalar function.
In particular, we consider

.. math::

    f(x,y) = (x - 0.5)^2 + (y - 0.5)^2 - 0.64\,.

The domain that minimizes this shape functional is a
disc of radius :math:`0.8` centered at :math:`(0.5,0.5)`.


Import modules
^^^^^^^^^^^^^^

We begin by importing Firedrake, Fireshape, and ROL. ::

    from firedrake import *
    from fireshape import *
    import ROL

Implement the shape function
^^^^^^^^^^^^^^^^^^^^^^^^^^^^
To implement the shape function :math:`\mathcal{J}`, we use Fireshape's class
:bash:`PDEconstrainedObjective`. This requires specifying how to evaluate
:math:`\mathcal{J}` in the method
:bash:`PDEconstrainedObjective.objective_value`. (Although :math:`\mathcal{J}`
is not technically constrained to a boundary value problem, it is
convenient to use the class  :bash:`PDEconstrainedObjective` as this
automatically returns :bash:`NaN` on poor quality meshes.) ::

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

Select initial guess, control space, and inner product
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
We select a unit disk centered at the origin as initial domain.
To modify the domain, we create a control space of geometric
transformations discretized using finite elements. To compute
descent directions, we employ Riesz representatives of shape
derivatives with respect to a full :math:`H^1`-inner product.
With these, we create a control variable :bash:`q` that will
be updated by the optimization algorithm. ::

    mesh = UnitDiskMesh(refinement_level=3)
    Q = FeControlSpace(mesh)
    IP = H1InnerProduct(Q)
    q = ControlVector(Q, IP)

Instantiate objective function J
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
We instantiate :math:`J(\Omega)` using the class
:bash:`LevelsetFunction` we have created. During instantiation,
we also pass a call back function :bash:`cb` that stores the
shape iterates whenever :bash:`J` is evaluated. ::

    out = VTKFile("levelset_domain.pvd")
    J = LevelsetFunction(Q, cb=lambda: out.write(Q.mesh_m.coordinates))

Select the optimization algorithm and solve the problem
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Finally, we select a trust-region optimization algorithm with l-BFGS Hessian
updates and set the optimization stopping criteria in the dictionary
:bash:`pd`.  This, together with :bash:`J` and :bash:`q` are passed to ROL,
which solves the problem. ::

    pd = {'Step': {'Type': 'Trust Region'},
          'General':  {'Secant': {'Type': 'Limited-Memory BFGS',
                                           'Maximum Storage': 25}},
           'Status Test': {'Gradient Tolerance': 1e-3,
                           'Step Tolerance': 1e-8,
                           'Iteration Limit': 10}}
    params = ROL.ParameterList(pd, "Parameters")
    problem = ROL.OptimizationProblem(J, q)
    solver = ROL.OptimizationSolver(problem, params)
    solver.solve()

Result
^^^^^^
Typing :bash:`python3 levelset.py` in the terminal returns:

.. code-block:: none

    Dogleg Trust-Region Solver with Limited-Memory BFGS Hessian Approximation
      iter  value          gnorm          snorm          delta          #fval     #grad     tr_flag   
      0     1.126112e+00   3.490688e+00                  3.490688e+00   
      1     1.126112e+00   3.490688e+00   3.490688e+00   6.346706e-01   3         1         5         
      2     -3.020956e-01  1.233637e+00   6.346706e-01   6.346706e-01   4         2         0         
      3     -5.938990e-01  4.248237e-01   3.599971e-01   1.586676e+00   5         3         0         
      4     -6.400550e-01  8.740918e-02   1.923642e-01   3.966691e+00   6         4         0         
      5     -6.427868e-01  3.459086e-02   4.737737e-02   9.916727e+00   7         5         0         
      6     -6.433796e-01  5.010732e-03   3.173178e-02   2.479182e+01   8         6         0         
      7     -6.433957e-01  1.411214e-03   5.412392e-03   6.197955e+01   9         7         0         
      8     -6.433973e-01  3.455089e-04   2.050891e-03   1.549489e+02   10        8         0         
    Optimization Terminated with Status: Converged

We can inspect the result by opening the file :bash:`levelset_domain.pvd`
with `ParaView <https://www.paraview.org/>`_.
