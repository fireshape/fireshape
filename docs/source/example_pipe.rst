.. _example_pipe:
.. role:: bash(code)
   :language: bash

Example 3: Kinetic energy dissipation in a pipe
===============================================

In this example, we show how to minimize the shape functional

.. math::

    \mathcal{J}(\Omega) = \int_\Omega \nu \nabla \mathbf{u} : \nabla \mathbf{u} \,\mathrm{d}\mathbf{x}\,,

where :math:`\mathbf{u}:\mathbb{R}^d\to\mathbb{R^d}\,, d = 2,3,` is the velocity of an incompressible
fluid and :math:`\nu` is the fluid viscosity.
The fluid velocity :math:`\mathbf{u}` and the fluid pressure
:math:`p:\mathbb{R}^d\to\mathbb{R}` satisfy the incompressible Navier-Stokes
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

Initial domain
^^^^^^^^^^^^^^
For the 2D-example, the geometry of the initial domain is described in the following
`gmsh <http://gmsh.info/>`_ script (this has been tested with :bash:`gmsh v4.1.2`).

.. literalinclude:: ../../examples/pipe/pipe.geo
    :linenos:

The mesh can be generated typing :bash:`gmsh -2 -clscale 0.1 -format msh2 -o pipe.msh pipe.geo`
in the terminal.

For the 3D-example, the geometry of the initial domain is described in the following
`gmsh <http://gmsh.info/>`_ script.

.. literalinclude:: ../../examples/pipe/pipe3d.geo
    :linenos:

The mesh can be generated typing :bash:`gmsh -3 -clscale 0.2 -format msh2 -o pipe.msh pipe3d.geo`
in the terminal (this has been tested with :bash:`gmsh v4.1.2`).

Implementing the PDE constraint
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
We implement the boundary value problem that acts as PDE
constraint in a python module named :bash:`pipe_PDEconstraint.py`.
In the code, we highlight the lines which characterize
the weak formulation of this boundary value problem.

.. note::

    The Dirichlet boundary data :math:`\mathbf{g}` depends on the dimension :math:`d` (see *Lines 36-42*)

.. literalinclude:: ../../examples/pipe/pipe_PDEconstraint.py
    :emphasize-lines: 30,31, 43, 44
    :linenos:

.. note::

    The Navier-Stokes solver may fail to converge if
    too big an optimization step occurs in the optimization
    process. To address this issue, we set the state `u` to
    the previously computed solution whenever the state solver fails, and
    use a trust-region algorithm as optimization solver. This way,
    the trust-region method will notice that there is no improvement
    if the state solver fails and will thus reduce the trust-region radius.

Implementing the shape functional
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
We implement the shape functional :math:`\mathcal{J}`
in a python module named :bash:`pipe_objective.py`.
In the code, we highlight the lines
which characterize :math:`\mathcal{J}`.


.. literalinclude:: ../../examples/pipe/pipe_objective.py
    :emphasize-lines: 15,16,17,18
    :linenos:


Setting up and solving the problem
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
We set up the problem in the script :bash:`pipe_main.py`.

To set up the problem, we need to:

* load the mesh of the initial guess (*Line 10*),
* choose the discretization of the control
  (*Line 11*, Lagrangian finite elements of degree 1),
* choose the metric of the control space
  (*Line 12*, :math:`H^1`-seminorm with homogeneous Dirichlet boundary conditions
  on fixed boundaries),
* initialize the PDE contraint on the physical mesh :bash:`mesh_m` (*Line 16-23*),
  choosing different viscosity parameters depending on the physical dimension `dim`
* specify to save the function :math:`\mathbf{u}` after each iteration
  in the file :bash:`solution/u.pvd` by setting the function ``cb``
  appropriately (*Lines 26-28*),
* initialize the shape functional (*Line 31*),
  and the reduced shape functional (*Line 32*),
* add a regularization term to improve the mesh quality
  in the updated domains (*Lines 35-26*),
* specify the volume equality constraint (*Lines 39-42*)
* create a ROL optimization prolem (*Lines 45-50*),
  and solve it (*Line 60*). Note that the volume equality constraint
  is imposed in *Line 58*.

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
   :bash:`levels = [4, 3]`).
   These Bsplines lie in the box with lower left corner :math:`(1.5, 0)` and upper right corner :math:`(12., 6.)`
   (:bash:`bbox = [(1.5, 12.), (0, 6.)]`).
   With :bash:`boundary_regularities = [2, 0]` we prescribe that the transformation vanishes for :math:`x=1.5`
   and :math:`x=12` with :math:`C^1`-regularity, but it does not
   necessarily vanish for :math:`y=0` and :math:`y=6`. In light of this,
   we do not need to specify :bash:`fixed_bids` in the inner product.

   Using Bsplines to discretize the control leads to similar results.
   The corresponding implementation can be found in :bash:`examples/pipe/pipe_splines.py`.

.. literalinclude:: ../../examples/pipe/pipe_main.py
    :linenos:

Result
^^^^^^
For the 2D-example, typing :bash:`python3 pipe_main.py` in the terminal returns:

.. code-block:: none

    Augmented Lagrangian Solver
    Subproblem Solver: Trust Region
      iter  fval           cnorm          gLnorm         snorm          penalty   feasTol   optTol    #fval   #grad   #cval   subIter 
      0     4.390783e-01   0.000000e+00   2.032222e-01                  1.00e+01  1.26e-01  1.00e-02  
      1     3.462720e-01   3.240811e-01   3.067402e-01   1.563151e+00   1.00e+01  1.00e-01  2.03e-04  19      8       23      14      
      2     3.198133e-01   3.392975e-01   4.013867e-01   7.366692e-01   1.00e+02  1.26e-01  1.00e-01  42      25      61      20      
      3     3.148021e-01   3.113228e-02   1.526213e-01   1.051970e+00   1.00e+02  7.94e-02  1.00e-03  50      30      72      5       
      4     3.004597e-01   1.392743e-03   2.659601e-01   5.746658e-01   1.00e+02  5.01e-02  1.00e-04  73      48      111     20      
      5     2.929301e-01   5.021819e-04   2.247039e-01   2.724358e-01   1.00e+02  3.16e-02  1.00e-04  96      66      150     20      

Optimization Terminated with Status: Iteration Limit Exceeded
We can inspect the result by opening the file :bash:`u.pvd`
with `ParaView <https://www.paraview.org/>`_. We see that the
difference between the volume of the initial guess and of the
retrieved optimized design is roughly :math:`5\cdot 10^{-4}`.

For the 3D-example, typing :bash:`python3 pipe_main.py` in the terminal returns:

.. code-block:: none

    Do simulation and add output

We can inspect the result by opening the file :bash:`u.pvd`
with `ParaView <https://www.paraview.org/>`_. We see that the
difference between the volume of the initial guess and of the
retrieved optimized design is roughly :math:`2\cdot 10^{-3}`.
.. retrieved optimized design is roughly :math:`???`.

.. note::

   This problem can also be solved using Bsplines to discretize the control.
   For instance, one could replace *Line 11-12* with

   .. code-block:: none

        bbox = [(1.5, 12.), (0, 6.)]
        orders = [4, 4]
        levels = [4, 3]
        Q = fs.BsplineControlSpace(mesh, bbox, orders, levels, boundary_regularities=[2, 0])
        inner = fs.H1InnerProduct(Q)

   In this case, the control is discretized using tensorized cubic (:bash:`order = [4, 4]`) Bsplines
   (roughly :math:`2^4` in the :math:`x`-direction :math:`\times\, 2^3` in the :math:`y`-direction;
   :bash:`levels = [4, 3]`).
   These Bsplines lie in the box with lower left corner :math:`(1.5, 0)` and upper right corner :math:`(12., 6.)`
   (:bash:`bbox = [(1.5, 12.), (0, 6.)]`).
   With :bash:`boundary_regularities = [2, 0]` we prescribe that the transformation vanishes for :math:`x=1.5`
   and :math:`x=12` with :math:`C^1`-regularity, but it does not
   necessarily vanish for :math:`y=0` and :math:`y=6`. In light of this,
   we do not need to specify :bash:`fixed_bids` in the inner product.

   Using Bsplines to discretize the control leads to similar results.
   The corresponding implementation can be found in :bash:`examples/pipe/pipe_splines.py`.
