.. _levelset:
.. role:: bash(code)
   :language: bash

Example 1: Level Set
====================

In this example, we show how to minimize the shape functional

.. math::

    \mathcal{J}(\Omega) = \int_\Omega f(\mathbf{x}) \,\mathrm{d}\mathbf{x}\,.

where :math:`f:\mathbb{R}^2\to\mathbb{R}` is a scalar function.
In particular, we consider :math:`f(x,y) = (x^2 - 1)(y^2-1)`.

Implementating the shape functional
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
We implement the shape functional :math:`\mathcal{J}`
in a python module named :bash:`levelsetfunctional.py`.
In the code, we highlight the lines
which characterize :math:`\mathcal{J}`.


.. literalinclude:: levelsetfunctional.py
    :emphasize-lines: 13,17
    :linenos:

Setting up and solving the problem
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
To set up the problem, we need to:

* construct the mesh of the initial guess
  (*Line 7*, a square with lower left corner in the origin and edge lenght 1)
* choose the discretization of the control
  (*Line 8*, Lagrangian finite elements of degree 1)
* choose the metric of the control space
  (*Line 9*, :math:`H^1`-seminorm)

Then, we specify to save the mesh after each iteration
in the file :bash:`domain.pvd`
by setting the function ``cb`` appropriately (*Lines 13-14*).

Finally, we initilize the shape functional (*Line 17*),
create a ROL optimization prolem (*Lines 20-36*),
and solve it.

.. literalinclude:: levelset.py
    :linenos:

Result
^^^^^^

We can inspect the result by opening the file :bash:`domain.pvd`
with `ParaView <https://www.paraview.org/>`_.
