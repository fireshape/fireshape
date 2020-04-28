.. Fireshape documentation master file, created by
   sphinx-quickstart on Tue Jul 10 17:06:48 2018.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.
.. title:: Fireshape
.. role:: bash(code)
   :language: bash

Introduction
============

.. toctree::
   :maxdepth: 2
   :hidden:

   self
   installation
   example_levelset
   example_L2tracking
   example_pipe
   ROL

Overview
^^^^^^^^
Fireshape is a shape optimization toolbox for the finite
element library `Firedrake <https://www.firedrakeproject.org/>`_.

To set up a shape optimization problem, all you need to
provide is the mesh on an initial guess,
the shape functional, and the weak-form of its PDE-contraint.

Fireshape computes adjoints and assembles first and
second derivatives for you using
`pyadjoint <http://www.dolfin-adjoint.org/en/release/>`_,
and it solves the optimization problem using the rapid optimization library
(`ROL <https://trilinos.org/packages/rol/>`_).

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
contact. You can do so by sending an email to :bash:`a.paganini@leicester.ac.uk`
or to :bash:`wechsung@nyu.edu`.

You can find information on how to install Fireshape on the page :ref:`installation`.

On the page :ref:`example_levelset`, we show how to solve a toy shape optimization problem.

On the page :ref:`example_L2tracking`, we show how to solve a shape optimization problem
constrained to a linear boundary value problem.

On the page :ref:`example_pipe`, we show how to solve a shape optimization problem
constrained to a nonlinear boundary value problem and a volume constraint.

On the page :ref:`ROL`, we give a very brief introduction to ROL.


