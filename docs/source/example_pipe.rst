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

The mesh can be generated typing :bash:`gmsh -2 -clscale 0.05 pipe.geo`
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

.. todo::

   Shall we use splines??

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

.. todo::

   Add details about imposing volume constraint

* create a ROL optimization prolem (*Lines 42-65*),
  and solve it (*Line 66*).

.. literalinclude:: ../../examples/pipe/pipe_main.py
    :linenos:

Result
^^^^^^
Typing :bash:`python3 pipe_main.py` in the terminal returns:

.. todo::

   Fix code and include output


.. code-block:: none

    ?????

We can inspect the result by opening the file :bash:`u.pvd`
with `ParaView <https://www.paraview.org/>`_.
