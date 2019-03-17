.. _example_L2tracking:
.. role:: bash(code)
   :language: bash

Example 2: L2-tracking
======================

In this example, we show how to minimize the shape functional

.. math::

    \mathcal{J}(\Omega) = \int_\Omega \big(u(\mathbf{x}) - u_t(\mathbf{x})\big)^2 \,\mathrm{d}\mathbf{x}\,.

where :math:`u:\mathbb{R}^2\to\mathbb{R}` is the solution to the scalar boundary value problem

.. math::

    -\Delta u = 4 \quad \text{in }\Omega\,, \qquad u = 0 \quad \text{on } \partial\Omega


and :math:`u_t:\mathbb{R}^2\to\mathbb{R}` is a target function.
In particular, we consider

.. math::

    u_t(x,y) = 0.36 - (x - 0.5)^2 + (y - 0.5)^2 - 0.5\,.

Beside the empty set, the domain that minimizes this shape functional is a
disc of radius :math:`0.6` centered at :math:`(0.5,0.5)`.


Implementing the PDE constraint
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
We implement the boundary value problem that act as PDE
constraint in a python module named :bash:`L2tracking_PDEconstraint.py`.
In the code, we highlight the lines which characterize
the weak formulation of this boundary value problem.

.. literalinclude:: ../../examples/L2tracking/L2tracking_PDEconstraint.py
    :emphasize-lines: 22,23
    :linenos:

.. Note:: 

    To solve the discretized variational problem,
    we use **CG** with a multigrid preconditioner
    (see :bash:`self.params` in *Lines 27-35*).
    For 2D problems, one can also use direct solvers.

Implementing the shape functional
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
We implement the shape functional :math:`\mathcal{J}`
in a python module named :bash:`L2tracking_objective.py`.lines
which characterize :math:`\mathcal{J}`.


.. literalinclude:: ../../examples/L2tracking/L2tracking_objective.py
    :emphasize-lines: 3,13,18
    :linenos:

Setting up and solving the problem
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
We set up the problem in the script
:bash:`L2tracking_main.py`.

To set up the problem, we

* construct the mesh of the initial guess
  (*Line 8*, a unit square with lower left corner in the origin)
* choose the discretization of the control
  (*Line 9*, Lagrangian finite elements of degree 1)
* choose the metric of the control space
  (*Line 10*, based on linear elasticity energy norm)
* initialize the PDE contraint on the physical mesh :bash:`mesh_m` (*Line 15*)
* specify to save the function :math:`u` after each iteration
  in the file :bash:`u.pvd` by setting the function ``cb``
  appropriately (*Lines 18-21*).
* initialize the shape functional (*Line 24*),
  and the reduce shape functional (*Line 25*),
* create a ROL optimization prolem (*Lines 28-50*),
  and solve it (*Line 51*).

.. literalinclude:: ../../examples/L2tracking/L2tracking_main.py
    :linenos:

Result
^^^^^^
Typing :bash:`python3 L2tracking_main.py` in the terminal returns:

.. code-block:: none

    Dogleg Trust-Region Solver with Limited-Memory BFGS Hessian Approximation
      iter  value          gnorm          snorm          delta          #fval     #grad     tr_flag
      0     3.984416e-03   4.253880e-02                  4.253880e-02
      1     2.380406e-03   3.243635e-02   4.253880e-02   1.063470e-01   3         2         0
      2     8.449408e-04   1.731817e-02   1.063470e-01   1.063470e-01   4         3         0
      3     4.944712e-04   8.530990e-03   2.919058e-02   2.658675e-01   5         4         0
      4     1.783406e-04   5.042658e-03   5.182347e-02   6.646687e-01   6         5         0
      5     2.395524e-05   1.342447e-03   5.472223e-02   1.661672e+00   7         6         0
      6     6.994156e-06   4.090116e-04   2.218345e-02   4.154180e+00   8         7         0
      7     4.011765e-06   2.961338e-04   1.009231e-02   1.038545e+01   9         8         0
      8     1.170509e-06   2.423493e-04   1.715142e-02   2.596362e+01   10        9         0
      9     1.170509e-06   2.423493e-04   1.372960e-02   3.432399e-03   11        9         2
      10    1.085373e-06   4.112702e-04   3.432399e-03   3.432399e-03   12        10        0
      11    7.428449e-07   1.090433e-04   3.432399e-03   8.580998e-03   13        11        0
      12    6.755854e-07   8.358801e-05   2.034473e-03   2.145249e-02   14        12        0
    Optimization Terminated with Status: Converged

We can inspect the result by opening the file :bash:`u.pvd`
with `ParaView <https://www.paraview.org/>`_.
