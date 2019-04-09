.. _example_pipe:
.. role:: bash(code)
   :language: bash

Example 3: Kinetic energy dissipation in a pipe
===============================================

In this example, we show how to minimize the shape functional

.. math::

    \mathcal{J}(\Omega) = \int_\Omega \nu \nabla \mathbf{u} : \nabla \mathbf{u} \,\mathrm{d}\mathbf{x}\,,

where :math:`\mathbf{u}:\mathbb{R}^2\to\mathbb{R^2}` is the velocity of an incompressible
fluid and :math:`\nu` is the fluid viscosity.
The fluid velocity :math:`\mathbf{u}` and the fluid pressure
:math:`p:\mathbb{R}^2\to\mathbb{R}` satisfy the incompressible Navier-Stokes
equations

.. math::
    :nowrap:

    \begin{align*}
    -\nu \Delta \mathbf{u} + \mathbf{u} \nabla \mathbf{u} + \nabla p &= 0 & \text{in } \Omega\,,\\
    \operatorname{div} \mathbf{u} &= 0& \text{in } \Omega\,,\\
    \mathbf{u} &= \mathbf{g} &\text{on } \partial\Omega\setminus \Gamma\,,\\
    p\mathbf{n} - \nu \nabla u\cdot \mathbf{n} & = 0 &\text{on } \Gamma\,.
    \end{align*}

Here, :math:`\mathbf{g}` is given by a Poiseuille flow at the inlet and is zero on the walls of the pipe.
The letter :math:`\Gamma` denotes the outlet.

In addition to the PDE-contstraint, we enforce a volume constraint:
the volume of the domain should remain constant during the optimization process.


The geometry of the initial domain is described in the following
`gmsh <http://gmsh.info/>`_ script.

.. literalinclude:: ../../examples/pipe/pipe.geo
    :linenos:

The mesh can be generated typing :bash:`gmsh -2 -clscale 0.1 pipe.geo`
in the terminal.


Implementing the PDE constraint
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
We implement the boundary value problem that acts as PDE
constraint in a python module named :bash:`pipe_PDEconstraint.py`.
In the code, we highlight the lines which characterize
the weak formulation of this boundary value problem.

.. literalinclude:: ../../examples/pipe/pipe_PDEconstraint.py
    :emphasize-lines: 28,29,34,35
    :linenos:



Implementing the shape functional
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
We implement the shape functional :math:`\mathcal{J}`
in a python module named :bash:`pipe_objective.py`.
In the code, we highlight the lines
which characterize :math:`\mathcal{J}`.


.. literalinclude:: ../../examples/pipe/pipe_objective.py
    :emphasize-lines: 13,14,15,16
    :linenos:


Setting up and solving the problem
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
We set up the problem in the script :bash:`pipe_main.py`.

To set up the problem, we need to:

* load the mesh of the initial guess
  (*Line 8*)
* choose the discretization of the control
  (*Line 9*, Lagrangian finite elements of degree 1)

.. note::

   This problem can also be solved using Bsplines to discretize the control.
   For instance, one could replace *Line 9-10* with

   .. code-block:: none

        bbox = [(1.5, 12.), (0, 6.)]
        orders = [4, 4]
        levels = [4, 3]
        Q = fs.BsplineControlSpace(mesh, bbox, orders, levels, boundary_regularities=[2, 0])
        inner = fs.H1InnerProduct(Q)

   In this case, the control is discretized using tensorized cubic (:bash:`order = [4, 4]`) Bsplines
   (roughly :math:`2^4` in the :math:`x`-direction :math:`\times\, 2^3` in the :math:`y`-direction;
   :bash:``levels = [4, 3]`).
   These Bsplines lie in the box with lower left corner :math:`(1.5, 0)` and upper right corner :math:`(12., 6.)`
   (:bash:`bbox = [(1.5, 12.), (0, 6.)]`).
   With :bash:`boundary_regularities = [2, 0]` we prescribe that the transformation vanishes for :math:`x=1.5`
   and :math:`x=12` with :math:`C^1`-regularity, but it does not
   necessarily vanish for :math:`y=0` and :math:`y=6`. In light of this,
   we do not need to specify :bash:`fixed_bids` in the inner product.

   Using Bsplines to discretize the control leads to similar results.

* choose the metric of the control space
  (*Line 10*, :math:`H^1`-seminorm with homogeneous Dirichlet boundary conditions
  on fixed boundaries)
* initialize the PDE contraint on the physical mesh :bash:`mesh_m` (*Line 14*)
* specify to save the function :math:`\mathbf{u}` after each iteration
  in the file :bash:`u.pvd` by setting the function ``cb``
  appropriately (*Lines 18-21*).
* initialize the shape functional (*Line 23*),
  and the reduced shape functional (*Line 24*),
* specify the volume equality constraint (*Lines 27-40*)
* create a ROL optimization prolem (*Lines 42-65*),
  and solve it (*Line 66*). Note that the volume equality constraint
  is imposed in *Line 64*.

.. literalinclude:: ../../examples/pipe/pipe_main.py
    :linenos:

Result
^^^^^^
Typing :bash:`python3 pipe_main.py` in the terminal returns:

.. code-block:: none

    Augmented Lagrangian Solver
    Subproblem Solver: Line Search
      iter  fval           cnorm          gLnorm         snorm          penalty   feasTol   optTol    #fval   #grad   #cval   subIter
      0     6.122697e-01   0.000000e+00   4.803481e-01                  1.00e+01  1.26e-01  4.80e-03

    Quasi-Newton Method with Limited-Memory BFGS
    Line Search: Cubic Interpolation satisfying Strong Wolfe Conditions
      iter  value          gnorm          snorm          #fval     #grad     ls_#fval  ls_#grad
      0     6.122697e-01   4.803481e-01
      1     5.090272e-01   2.231151e-01   4.803481e-01   2         2         1         0
      2     4.675826e-01   1.518620e-01   2.484078e-01   3         3         1         0
      3     4.182690e-01   1.340299e-01   7.702604e-01   4         4         1         0
      4     4.124342e-01   1.285919e-01   2.800089e-01   5         5         1         0
      5     4.052552e-01   4.943457e-02   1.032895e-01   6         6         1         0
    Optimization Terminated with Status: Iteration Limit Exceeded
      1     3.941314e-01   5.711779e-01   1.893038e-01   1.372841e+00   1.05e+01  1.26e-01  1.00e-01  9       8       13      5

    Quasi-Newton Method with Limited-Memory BFGS
    Line Search: Cubic Interpolation satisfying Strong Wolfe Conditions
      iter  value          gnorm          snorm          #fval     #grad     ls_#fval  ls_#grad
      0     5.109319e-01   1.374670e+00
      1     4.180459e-01   5.268006e-02   1.374670e-01   3         2         2         0
    Optimization Terminated with Status: Converged
      2     4.166614e-01   6.218581e-02   2.017320e-01   3.079778e-01   1.05e+01  9.95e-02  9.52e-03  13      10      17      1

    Quasi-Newton Method with Limited-Memory BFGS
    Line Search: Cubic Interpolation satisfying Strong Wolfe Conditions
      iter  value          gnorm          snorm          #fval     #grad     ls_#fval  ls_#grad
      0     4.208149e-01   1.491077e-01
      1     4.196027e-01   4.684224e-02   1.629614e-02   3         2         2         0
      2     4.193933e-01   4.351257e-02   4.854066e-03   4         3         1         0
      3     4.176366e-01   1.579725e-02   7.989770e-02   5         4         1         0
      4     4.173411e-01   1.195939e-02   2.854034e-02   6         5         1         0
      5     4.170438e-01   6.051141e-03   6.532997e-02   7         6         1         0
    Optimization Terminated with Status: Converged
      3     4.164821e-01   1.154438e-02   2.317212e-02   1.939492e-01   1.05e+01  7.87e-02  9.07e-04  21      16      29      5

    Quasi-Newton Method with Limited-Memory BFGS
    Line Search: Cubic Interpolation satisfying Strong Wolfe Conditions
      iter  value          gnorm          snorm          #fval     #grad     ls_#fval  ls_#grad
      0     4.171393e-01   3.202939e-02
      1     4.170869e-01   5.706707e-03   3.270210e-03   3         2         2         0
      2     4.170838e-01   5.419623e-03   5.735988e-04   4         3         1         0
      3     4.170420e-01   2.061273e-03   1.464455e-02   5         4         1         0
      4     4.170337e-01   1.800966e-03   6.031019e-03   6         5         1         0
      5     4.170109e-01   2.404397e-03   2.167103e-02   7         6         1         0
    Optimization Terminated with Status: Iteration Limit Exceeded
      4     4.169879e-01   4.340375e-04   9.207352e-03   4.088152e-02   1.05e+01  6.22e-02  8.64e-05  29      22      41      5

    Quasi-Newton Method with Limited-Memory BFGS
    Line Search: Cubic Interpolation satisfying Strong Wolfe Conditions
      iter  value          gnorm          snorm          #fval     #grad     ls_#fval  ls_#grad
      0     4.170110e-01   2.778616e-03
      1     4.170097e-01   4.301760e-03   9.921214e-04   3         2         2         0
      2     4.170078e-01   3.938541e-03   8.334239e-04   4         3         1         0
      3     4.169930e-01   2.105004e-03   1.183851e-02   5         4         1         0
      4     4.169876e-01   2.289890e-03   5.421771e-03   6         5         1         0
      5     4.169613e-01   4.067340e-03   2.904419e-02   7         6         1         0
    Optimization Terminated with Status: Iteration Limit Exceeded
      5     4.169267e-01   6.476467e-04   1.557539e-02   4.814567e-02   1.05e+01  4.92e-02  8.23e-06  37      28      53      5

    Quasi-Newton Method with Limited-Memory BFGS
    Line Search: Cubic Interpolation satisfying Strong Wolfe Conditions
      iter  value          gnorm          snorm          #fval     #grad     ls_#fval  ls_#grad
      0     4.169616e-01   5.250610e-03
      1     4.169594e-01   4.103997e-03   8.471029e-04   3         2         2         0
      2     4.169577e-01   3.953634e-03   5.215962e-04   4         3         1         0
      3     4.169292e-01   1.663547e-03   1.651461e-02   5         4         1         0
      4     4.169212e-01   2.091883e-03   7.624444e-03   6         5         1         0
      5     4.168917e-01   3.977605e-03   3.308807e-02   7         6         1         0
    Optimization Terminated with Status: Iteration Limit Exceeded
      6     4.169006e-01   1.657174e-04   1.523176e-02   5.589063e-02   1.05e+01  3.89e-02  1.00e-06  45      34      65      5
    Optimization Terminated with Status: Iteration Limit Exceeded
    -0.00016571737260306918


We can inspect the result by opening the file :bash:`u.pvd`
with `ParaView <https://www.paraview.org/>`_. We see that the
difference between the volume of the initial guess and of the
retrieved optimized design is less than :math:`1.7\cdot 10^{-4}`.
