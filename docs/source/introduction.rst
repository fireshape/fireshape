Introduction
============

Overview
^^^^^^^^
Fireshape is a shape optimization toolbox for the finite
element library `Firedrake <https://www.firedrakeproject.org/>`_.

To set up a shape optimization problem, all you need to
provide is the mesh on an initial guess,
the shape functional, and the weak-form of its PDE-contraint.

Fireshape computes adjoints and assembles first and
second derivatives for you, and it solves the optimization
problem using the rapid optimization library (`ROL <https://trilinos.org/packages/rol/>`_).

Features
^^^^^^^^
Fireshape neatly distinguishes between the discretization
of control and state variables.
To discretize the control, you can choose among finite elements,
B-splines, and the free-form approach.
To discretize the state, you have access to all finite element
spaces offered by Firedrake.


Fireshape relies on the mesh-deformation approach to update the
geometry of the domain. By specifying the metric of the control
space, you can decide whether meshes should be updated using
Laplace or elasticity equations. In 2D, you can also use the elasticity
equation corrected with Cauchy-Riemann terms, which generally leads
to very high-quality meshes.

Where to go from here
^^^^^^^^^^^^^^^^^^^^^
If you are interested in using Fireshape, do not hesitate to get in
contact. You can do so by using the contact forms available
`here <https://www.maths.ox.ac.uk/people/alberto.paganini/contact>`__
or `here <https://www.maths.ox.ac.uk/people/florian.wechsung/contact>`__.

You can find information on how to install Fireshape on the page :ref:`installation`.

On the page :ref:`example_levelset`, we show how to solve a toy shape optimization problem.

On the page :ref:`example_L2tracking`, we show how to solve a shape optimization problem
constrained to a boundary value problem.
