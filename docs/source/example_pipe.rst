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
   The corresponding implementation can be found in :bash:`examples/pipe/pipe_splines.py`.

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

.. note::

    This shape optimization problem is not easy. In particular,
    it is not trivial to find a good combination of the optimization algorithm parameters
    (e.g., :bash:`Penalty Parameter Growth Factor` and :bash:`Status Test`).
    In particular, an unfortunate choice can lead to self-intersecting meshes.
    This issue will be resolved in the future (cf. `this github-issue <https://github.com/fireshape/fireshape/issues/6>`_).

.. literalinclude:: ../../examples/pipe/pipe_main.py
    :linenos:

Result
^^^^^^
For the 2D-example, typing :bash:`python3 pipe_main.py` in the terminal returns:

.. code-block:: none

     Augmented Lagrangian Solver
    Subproblem Solver: Line Search
      iter  fval           cnorm          gLnorm         snorm          penalty   feasTol   optTol    #fval   #grad   #cval   subIter 
      0     6.114543e-01   0.000000e+00   4.853678e-01                  1.00e+01  1.26e-01  1.00e-02  
    
    Quasi-Newton Method with Limited-Memory BFGS
    Line Search: Cubic Interpolation satisfying Strong Wolfe Conditions
      iter  value          gnorm          snorm          #fval     #grad     ls_#fval  ls_#grad  
      0     6.114543e-01   4.853678e-01   
      1     5.082162e-01   2.242712e-01   4.853678e-01   2         2         1         0         
      2     4.705059e-01   1.492113e-01   2.250649e-01   3         3         1         0         
      3     4.250298e-01   1.178931e-01   6.810325e-01   4         4         1         0         
      4     4.179713e-01   9.755194e-02   2.343119e-01   5         5         1         0         
      5     4.132288e-01   3.829923e-02   7.983724e-02   6         6         1         0         
    Optimization Terminated with Status: Iteration Limit Exceeded
      1     4.010445e-01   5.891798e-01   1.445512e-01   1.305843e+00   1.04e+01  1.26e-01  1.00e-01  9       8       13      5       
    
    Quasi-Newton Method with Limited-Memory BFGS
    Line Search: Cubic Interpolation satisfying Strong Wolfe Conditions
      iter  value          gnorm          snorm          #fval     #grad     ls_#fval  ls_#grad  
      0     5.277617e-01   1.417356e+00   
      1     4.281330e-01   3.716341e-02   1.417356e-01   3         2         2         0         
    Optimization Terminated with Status: Converged
      2     4.262437e-01   7.194284e-02   1.402643e-01   3.399747e-01   1.04e+01  9.96e-02  9.62e-03  13      10      17      1       
    
    Quasi-Newton Method with Limited-Memory BFGS
    Line Search: Cubic Interpolation satisfying Strong Wolfe Conditions
      iter  value          gnorm          snorm          #fval     #grad     ls_#fval  ls_#grad  
      0     4.319118e-01   1.838819e-01   
      1     4.301684e-01   3.412084e-02   1.898993e-02   3         2         2         0         
      2     4.300566e-01   3.178969e-02   3.450143e-03   4         3         1         0         
      3     4.290978e-01   9.833390e-03   5.853338e-02   5         4         1         0         
      4     4.289745e-01   6.990576e-03   1.891007e-02   6         5         1         0         
    Optimization Terminated with Status: Converged
      3     4.285971e-01   6.857941e-03   2.638424e-02   1.006545e-01   1.04e+01  7.88e-02  9.25e-04  20      15      27      4       
    
    Quasi-Newton Method with Limited-Memory BFGS
    Line Search: Cubic Interpolation satisfying Strong Wolfe Conditions
      iter  value          gnorm          snorm          #fval     #grad     ls_#fval  ls_#grad  
      0     4.290088e-01   1.997817e-02   
      1     4.289863e-01   7.039874e-03   2.246597e-03   3         2         2         0         
      2     4.289815e-01   6.761098e-03   7.469821e-04   4         3         1         0         
      3     4.288771e-01   4.462582e-03   2.782104e-02   5         4         1         0         
      4     4.288394e-01   3.198473e-03   1.830970e-02   6         5         1         0         
      5     4.288063e-01   2.825638e-03   2.403640e-02   7         6         1         0         
    Optimization Terminated with Status: Iteration Limit Exceeded
      4     4.287463e-01   1.035550e-03   1.066469e-02   6.928764e-02   1.04e+01  6.24e-02  1.00e-04  28      21      39      5       
    Optimization Terminated with Status: Step Tolerance Met
    0.0010355498010312658

We can inspect the result by opening the file :bash:`u.pvd`
with `ParaView <https://www.paraview.org/>`_. We see that the
difference between the volume of the initial guess and of the
retrieved optimized design is roughly :math:`1\cdot 10^{-3}`.


For the 3D-example, typing :bash:`python3 pipe_main.py` in the terminal returns:

.. code-block:: none

     Augmented Lagrangian Solver
    Subproblem Solver: Line Search
      iter  fval           cnorm          gLnorm         snorm          penalty   feasTol   optTol    #fval   #grad   #cval   subIter 
      0     1.347651e+01   0.000000e+00   1.000000e+00                  1.71e+01  7.76e-02  1.00e-02  
    
    Quasi-Newton Method with Limited-Memory BFGS
    Line Search: Cubic Interpolation satisfying Strong Wolfe Conditions
      iter  value          gnorm          snorm          #fval     #grad     ls_#fval  ls_#grad  
      0     1.711067e+00   1.000000e+00   
      1     1.642542e+00   9.963956e-01   1.000000e+00   2         2         1         0         
      2     1.417121e+00   2.175011e-01   4.272211e-01   3         3         1         0         
      3     1.394307e+00   1.608342e-01   1.235115e-01   4         4         1         0         
      4     1.351738e+00   1.394124e-01   4.823868e-01   5         5         1         0         
      5     1.343115e+00   4.789544e-02   1.445996e-01   6         6         1         0         
    Optimization Terminated with Status: Iteration Limit Exceeded
      1     9.932418e+00   1.606219e+00   3.772286e-01   1.172752e+00   1.78e+01  7.76e-02  5.84e-02  9       8       13      5       
    
    Quasi-Newton Method with Limited-Memory BFGS
    Line Search: Cubic Interpolation satisfying Strong Wolfe Conditions
      iter  value          gnorm          snorm          #fval     #grad     ls_#fval  ls_#grad  
      0     2.720814e+00   7.340730e+00   
      1     1.491105e+00   3.804867e-01   3.287054e-01   4         2         3         0         
      2     1.487290e+00   6.000800e-02   1.798864e-02   5         3         1         0         
      3     1.487128e+00   4.471237e-02   3.609876e-03   6         4         1         0         
    Optimization Terminated with Status: Converged
      2     1.158919e+01   1.665100e-01   3.521585e-01   1.096183e+00   1.78e+01  5.82e-02  3.28e-03  16      12      22      3       
    
    Quasi-Newton Method with Limited-Memory BFGS
    Line Search: Cubic Interpolation satisfying Strong Wolfe Conditions
      iter  value          gnorm          snorm          #fval     #grad     ls_#fval  ls_#grad  
      0     1.518502e+00   7.408525e-01   
      1     1.514873e+00   6.394189e-01   7.408525e-02   3         2         2         0         
      2     1.503780e+00   4.942620e-02   3.403998e-02   4         3         1         0         
      3     1.503651e+00   4.662890e-02   2.723661e-03   5         4         1         0         
      4     1.502176e+00   8.098043e-02   4.938122e-02   6         5         1         0         
      5     1.501550e+00   5.914622e-02   3.359007e-02   7         6         1         0         
    Optimization Terminated with Status: Iteration Limit Exceeded
      3     1.179458e+01   2.017731e-02   4.658408e-01   1.880643e-01   1.78e+01  4.37e-02  1.85e-04  24      18      34      5       
    
    Quasi-Newton Method with Limited-Memory BFGS
    Line Search: Cubic Interpolation satisfying Strong Wolfe Conditions
      iter  value          gnorm          snorm          #fval     #grad     ls_#fval  ls_#grad  
      0     1.502010e+00   1.438533e-01   
      1     1.501779e+00   1.219637e-01   1.438533e-02   3         2         2         0         
      2     1.501386e+00   2.732084e-02   6.165906e-03   4         3         1         0         
      3     1.501343e+00   2.587039e-02   1.663911e-03   5         4         1         0         
      4     1.500886e+00   4.926075e-02   2.793769e-02   6         5         1         0         
      5     1.500673e+00   4.101282e-02   1.867452e-02   7         6         1         0         
    Optimization Terminated with Status: Iteration Limit Exceeded
      4     1.182298e+01   2.140905e-03   3.230205e-01   6.098945e-02   1.78e+01  3.27e-02  1.00e-04  32      24      46      5       
    Optimization Terminated with Status: Step Tolerance Met
    -0.002140904667484733

We can inspect the result by opening the file :bash:`u.pvd`
with `ParaView <https://www.paraview.org/>`_. We see that the
difference between the volume of the initial guess and of the
retrieved optimized design is roughly :math:`2\cdot 10^{-3}`.
