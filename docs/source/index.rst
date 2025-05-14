.. Fireshape documentation master file, created by
   sphinx-quickstart on Tue Jul 10 17:06:48 2018.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.
.. title:: Fireshape
.. role:: bash(code)
   :language: bash

Introduction
============

Welcome to the documentation for Fireshape.

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
Laplace or elasticity equations.

Where to go from here
^^^^^^^^^^^^^^^^^^^^^
If you are interested in using Fireshape, do not hesitate to get in
contact. You can do so by sending an email to :bash:`a.paganini@leicester.ac.uk`.

You can find information on how to install Fireshape on the page :ref:`installation`.

On the page :ref:`example_levelset`, we show how to solve a toy shape optimization problem.

On the page :ref:`example_dido`, we show how to solve a `classic <https://en.m.wikipedia.org/wiki/Dido>`_ shape
optimization problem with perimeter constraint.

On the page :ref:`example_heat`, we show how to solve a shape optimization problem
constrained to a time-dependent linear boundary value problem.

On the page :ref:`ROL`, we give a very brief introduction to ROL.

Finally, you can find a manuscript about Fireshape `here <https://arxiv.org/abs/2005.07264>`_.

.. toctree::
   :maxdepth: 2
   :hidden:

   self
   installation
   example_levelset
   example_dido
   example_heat
   ROL
