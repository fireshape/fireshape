.. _example_heat:
.. role:: bash(code)
   :language: bash

Example 3: Heat equation
========================

In this example, we show how to minimize the shape function

.. math::

    J(\Omega) = \int_\Omega (u(T, \mathbf{x}) - u_t(\mathbf{x}))^2 \,\mathrm{d}\mathbf{x}\,,

where :math:`u_t:\mathbb{R}^3\to\mathbb{R}` is a target temperature profile and
:math:`u:[0,T]\times\mathbb{R}^3\to\mathbb{R}` is the solution to the heat equation

.. math::

    \partial_t u - \Delta u = f\quad \text{in }(0,T]\times\Omega\,,\\
    u(0,\cdot)=u_0 \quad \text{in }\Omega\,,\\
    u(\cdot,\mathbf{x})=0\quad \text{on }\partial\Omega\,.

In particular, we consider :math:`T=\pi/2`, 

.. math::

    u_t(x,y,z) = 0.64 - (x - 0.5)^2 - (y - 0.5)^2 - (z - 0.5)^2\,,

and

.. math::

    f(t,x,y,z) = 6\sin(t) + \cos(t)\, u_t(x, y, z)\,.

The domain that minimizes :math:`J(\Omega)` is a
ball of radius :math:`0.8` centered at :math:`(0.5,0.5,0.5)`.

In the following, we describe how to solve this problem in Fireshape.
The entire script is contained in the Python file 
:bash:`heat.py`, which is saved in the Fireshape repository and
can be found at the following link:
`link-to-heat-example <https://github.com/fireshape/fireshape/tree/master/examples/heat/heat.py>`_.

Import modules
^^^^^^^^^^^^^^

We begin by importing Firedrake, Fireshape, and ROL.

.. literalinclude:: ../../examples/heat/heat.py
    :lines: 1-3
    :linenos:
    :lineno-match:

Implement the shape function
^^^^^^^^^^^^^^^^^^^^^^^^^^^^
To implement the shape function :math:`J`, we use Fireshape's class
:bash:`PDEconstrainedObjective`. This requires specifying how to evaluate
:math:`J` in the method
:bash:`PDEconstrainedObjective.objective_value`. To solve the heat equation,
we use the implicit Euler method with 10 time steps.

.. literalinclude:: ../../examples/heat/heat.py
    :lines: 6-66
    :linenos:
    :lineno-match:


Select initial guess, control space, and inner product
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
We select a unit disk centered at the origin as initial domain.
To modify the domain, we create a control space of geometric
transformations discretized using finite elements. To compute
descent directions, we employ Riesz representatives of shape
derivatives with respect to a full :math:`H^1`-inner product.
With these, we create a control variable :bash:`q` that will
be updated by the optimization algorithm.

.. literalinclude:: ../../examples/heat/heat.py
    :lines: 69-73
    :linenos:
    :lineno-match:

Instantiate objective function J
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
We instantiate :math:`J(\Omega)` using the class
:bash:`TargetHeat` we have created. During instantiation,
we also pass a call back function :bash:`cb` that stores the
shape iterates whenever :bash:`J` is evaluated. We also store
the evolution of the temperature profile on the initial domain.

.. literalinclude:: ../../examples/heat/heat.py
    :lines: 75-78
    :linenos:
    :lineno-match:

Select the optimization algorithm and solve the problem
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Finally, we select a trust-region optimization algorithm with l-BFGS Hessian
updates and set the optimization stopping criteria in the dictionary
:bash:`pd`.  This, together with :bash:`J` and :bash:`q` are passed to ROL,
which solves the problem. When the optimization algorithms has stopped,
we store the evolution of the temperature profile
on the optimized domain.

.. literalinclude:: ../../examples/heat/heat.py
    :lines: 80-91
    :linenos:
    :lineno-match:

Result
^^^^^^
Typing :bash:`python3 levelset.py` in the terminal returns:

.. code-block:: none

    Storing the initial temperature evolution.

    Dogleg Trust-Region Solver with Limited-Memory BFGS Hessian Approximation
      iter  value          gnorm          snorm          delta          #fval     #grad     tr_flag   
      0     7.270316e+00   1.687694e+01                  1.687694e+01   
      1     7.270316e+00   1.687694e+01   1.687694e+01   3.068535e+00   3         1         5         
      2     7.270316e+00   1.687694e+01   3.068535e+00   3.341968e-01   4         1         5         
      3     3.166234e+00   8.472555e+00   3.341968e-01   3.341968e-01   5         2         0         
      4     1.193140e+00   3.822150e+00   3.341968e-01   8.354921e-01   6         3         0         
      5     4.522666e-01   1.745016e+00   2.762971e-01   2.088730e+00   7         4         0         
      6     1.692963e-01   7.689023e-01   2.341998e-01   5.221825e+00   8         5         0         
      7     7.025110e-02   3.313803e-01   1.875483e-01   1.305456e+01   9         6         0         
      8     3.602387e-02   1.456138e-01   1.508964e-01   3.263641e+01   10        7         0         
      9     1.990907e-02   9.035061e-02   1.493638e-01   8.159102e+01   11        8         0         
      10    4.282918e-03   4.940063e-02   2.828876e-01   2.039776e+02   12        9         0         
      11    2.436980e-03   3.761708e-02   7.220499e-02   5.099439e+02   13        10        0         
      12    1.130037e-04   6.559129e-03   1.380168e-01   1.274860e+03   14        11        0         
      13    7.578703e-05   3.833176e-03   1.429239e-02   1.274860e+03   15        12        0         
      14    6.393627e-05   2.059663e-03   4.817841e-03   3.187149e+03   16        13        0         
      15    5.661970e-05   1.580813e-03   6.664030e-03   7.967873e+03   17        14        0         
      16    5.173678e-05   1.058514e-03   6.662573e-03   1.991968e+04   18        15        0         
      17    4.982127e-05   3.074662e-04   4.933308e-03   4.979921e+04   19        16        0         
    Optimization Terminated with Status: Converged
    Storing the final temperature evolution.

We can inspect the result by opening the file :bash:`levelset_domain.pvd`
with `ParaView <https://www.paraview.org/>`_. In the GIF below, we see that
temperature evolution in the initial domain, the domain (black grid)
converging to the right shape (red ball), and the temperature evolution in
the optimized domain.

.. image:: ./_static/example_heat.gif
    :alt: Animated GIF created with Pillow
