.. _levelset:
.. role:: bash(code)
   :language: bash

Example 1: Level Set
====================

In this example, we show how to minimize the shape functional

.. math::

    \mathcal{J}(\Omega) = \int_\Omega f(\mathbf{x}) \,\mathrm{d}\mathbf{x}\,.

where :math:`f:\mathbb{R}^2\to\mathbb{R}` is a scalar function.
In particular, we consider

.. math::

    f(x,y) = (x - 0.5)^2 + (y - 0.5)^2 - 0.5\,.

The domain that minimizes this shape functional is a
disc of radius :math:`1/\sqrt{2}` centered at :math:`(0.5,0.5)`.


Implementing the shape functional
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
  (*Line 7*, a unit square with lower left corner in the origin)
* choose the discretization of the control
  (*Line 8*, Lagrangian finite elements of degree 1)
* choose the metric of the control space
  (*Line 9*, :math:`H^1`-seminorm)

Then, we specify to save the mesh after each iteration
in the file :bash:`domain.pvd`
by setting the function ``cb`` appropriately (*Lines 13-14*).

Finally, we initialize the shape functional (*Line 17*),
create a ROL optimization prolem (*Lines 20-36*),
and solve it (*Line 37*).

.. literalinclude:: levelset.py
    :linenos:

Result
^^^^^^
Typing :bash:`python3 levelset.py` in the terminal returns:

.. code-block:: none

    Quasi-Newton Method with Limited-Memory BFGS
    Line Search: Cubic Interpolation satisfying Strong Wolfe Conditions
      iter  value          gnorm          snorm          #fval     #grad     ls_#fval  ls_#grad
      0     -3.333333e-01  2.426719e-01
      1     -3.765250e-01  1.121533e-01   2.426719e-01   2         2         1         0
      2     -3.886767e-01  8.040797e-02   2.525978e-01   3         3         1         0
      3     -3.919733e-01  2.331695e-02   6.622577e-02   4         4         1         0
      4     -3.926208e-01  4.331993e-03   5.628606e-02   5         5         1         0
      5     -3.926603e-01  2.906313e-03   1.239420e-02   6         6         1         0
      6     -3.926945e-01  9.456530e-04   2.100085e-02   7         7         1         0
      7     -3.926980e-01  3.102278e-04   6.952015e-03   8         8         1         0
      8     -3.926987e-01  1.778454e-04   3.840828e-03   9         9         1         0
      9     -3.926989e-01  9.001788e-05   2.672387e-03   10        10        1         0
    Optimization Terminated with Status: Converged


We can inspect the result by opening the file :bash:`domain.pvd`
with `ParaView <https://www.paraview.org/>`_.
