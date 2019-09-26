.. _ROL:
.. role:: bash(code)
   :language: bash

Using ROL in Fireshape
======================

Fireshape allows solving shape optimization problems
using the `Rapid Optimization Library (ROL) <https://trilinos.org/packages/rol/>`_.
The goal of this page is to give a very brief introduction to ROL
and how to use it in Fireshape. Please note that this is not
an official guide, but it's merely based on reverse engineering
ROL's source code.

Basics
^^^^^^

ROL is an optimization library that allows a complete
mathematical description of the control space.
More specifically, ROL allows specifying vector space
operations like addition and scalar multiplication.
On top of that, ROL allows specifying which norm
endows the control space. Fireshape implements these methods
to describe a control space of geometric transformations.

To use ROL within Fireshape, you need to declare:

* the control vector :bash:`q` and the objective function :bash:`J`
  (see examples in this guide),
* a dictionary :bash:`params_dict` that specifies which algorithm to employ
  (see below).

With these objects, you can solve the optimization problem with the following code.

.. code-block:: none

    params = ROL.ParameterList(params_dict, "Parameters")
    problem = ROL.OptimizationProblem(J, q)
    solver = ROL.OptimizationSolver(problem, params)
    solver.solve()

If the optimization problem is subject to additional equality or inequality constraints,
you can include these by declaring:

* the equality constraint :bash:`econ` and its multiplier :bash:`emul`
* the inequality constraint :bash:`icon`, its multiplier :bash:`imul`, and the inequality bounds :bash:`ibnd`.

.. note::

    The variables :bash:`econ` and :bash:`icon` are of type :bash:`ROL.Constraint`,
    and can be instantiated usind the fireshape class :bash:`EqualityConstraint`.

    The variables :bash:`emul` and :bash:`imul` are of type :bash:`ROL.StdVector`.
    The variable :bash:`ibnd` is of type :bash:`ROL.BoundConstraint`. For example,
    the following code sets the lower and the upper bound to 2 and 7, respectively.

    .. code-block:: none

        lower = ROL.StdVector(1); lower[0] = 2
        upper = ROL.StdVector(1); upper[0] = 7
        ibnd = ROL.BoundConstraint(lower, upper)

With these objects, you can solve the optimization problem with the following code.

.. code-block:: none

    params = ROL.ParameterList(params_dict, "Parameters")
    problem = ROL.OptimizationProblem(J, q, econ=econ, emul=emul, icon=icon, imul=imul, ibnd=ibnd)
    solver = ROL.OptimizationSolver(problem, params)
    solver.solve()

.. note::

    ROL allows also specifying bound constraints on the control variable, that is,
    constraint of the form :bash:`a<q<b`. However, such constraints are not present
    in shape optimization problems modelled via geometric transformations.

Choosing optimization algorithms and setting parameters
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

You can select the optimization algorithm and set
optimization parameters by specifying the three
fields: :bash:`Step`, :bash:`General`, and :bash:`Status Test`
in the dictionary :bash:`params_dict`.

.. code-block:: none

    params_dict = {'Step': #set step parameters here,
                   'General': #set general parameters here,
                   'Status Test': #set status parameters here,
                  }

The simplest field to set is :bash:`Status Test`.
The understanding of the fields :bash:`Step` and :bash:`General`
is less immediate. In this guide, we restrict ourselves to two cases:
an unconstrained problem solved with a trust-region algorithm
and a constrained problem solved with the augmented Lagrangian
method.

.. note::

    We prefer using a trust-region method instead of a line-search
    method because in the former case it is easier to deal with
    situations when the routine that evaluates the functional :bash:`J`
    fails. Such failures are usually due to failure in solving the state
    constraint. Among other reasons, this can happen when the control
    :bash:`q` is not feasible (for instance, when the underlying mesh
    intersects itself) or when the state constraint is nonlinear and
    the optimization step is too large (in which case the initial guess
    is not good enough).

    In a nutshell, trust-region methods solve a sequence of optimization
    problems. In each of these, one minimizes a quadratic functional
    with control constraints. The idea is that the quadratic functional
    models the original objective functional. The control constraint limits the
    validity of this model to a trusted region. To construct the quadratic
    functional, one evaluates the original functional, its gradient,
    and its Hessian (or a BFGS approximation of it) in a feasible point.
    The minimizer to this quadratic functional is sought in a ball around
    that feasible point (computing this minimizer is computationally inexpensive
    and does not
    involve further evaluations of the original functional or its derivatives).
    Then, one evaluates the original objective functional in this minimizer
    and compares the *actual reduction* with the *predicted reduction*.
    The new control is accepted if the actual reduction is
    positive, that is, if there is actual reduction.
    Then, if there is good agreement between the actual and predicted
    reductions, the trust-region radius is increased. This radius is
    decreased if the actual reduction is not positive or the ratio between
    actual and predicted reductions is close to zero.
    From this, we understand that a safe solution to deal with failed
    evaluations of :bash:`J` is to store the previously computed value of
    :bash:`J` and, using a :bash:`try: ... except: ...` approach,
    return it if the new evaluation of :bash:`J` fails. This corresponds
    to a nonpositive actual reduction, which triggers a reduction of
    the trust-region radius.

.. note::

    The following examples include all parameters that can be set
    for the algorithms described. However, it is not necessary to
    specify a field if one does not want to modify a default value.

Setting termination criteria
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

This field sets the termination criteria of the algorithm.
Its use is self-explanatory. We report its syntax with the
default values.

.. code-block:: none

    'Status Test':{'Gradient Tolerance':1.e-6,
                   'Step Tolerance':1.e-12,
                   'Iteration Limit':100}

If the optimization problem contains equality (or inequality
constraints), one can further specify the desired
:bash:`Constraint Tolerance`. Its default value is :bash:`1.e-6`.
In this case, an optimization algorithm has converged only
if both the gradient and constraint tolerances are satisfied.

Solving an unconstrained problem with a trust-region method
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
To solve an unconstrained problem using a trust-region method,
we can set :bash:`Step` and :bash:`General` as follows
(the provided values are the default ones).
To understand some of these parameters, please read the trust-region algorithm
`implementation <https://github.com/trilinos/Trilinos/blob/master/packages/rol/src/step/trustregion/ROL_TrustRegion.hpp>`_.

.. code-block:: none

    'Step':{'Type':'Trust Region',
            'Trust Region':{'Initial Radius':-1, #determine initial radius with heuristics
                            'Maximum Radius':1.e8,
                            'Subproblem Solver':'Dogleg',
                            'Radius Growing Rate':2.5
                            'Step Acceptance Threshold':0.05,
                            'Radius Shrinking Threshold':0.05,
                            'Radius Growing Threshold':0.9,
                            'Radius Shrinking Rate (Negative rho)':0.0625,
                            'Radius Shrinking Rate (Positive rho)':0.25,
                            'Radius Growing Rate':2.5,
                            'Sufficient Decrease Parameter':1.e-4,
                            'Safeguard Size':100.0,
                           }
           }
    'General':{'Print Verbosity':0, #set to any number >0 for increased verbosity
               'Secant':{'Type':'Limited-Memory BFGS', #BFGS-based Hessian-update in trust-region model
                         'Maximum Storage':10
                        }
              }

Solving a constrained problem with an augmented Lagrangian method
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

To solve a problem with equality or inequality constraint
using an augmented Lagrangian method,
we can set :bash:`Step` and :bash:`General` as follows
(the provided values are the default ones).
Note that the augmented Lagrangian algorithm solves
a sequence of intermediate models. These intermediate models
are unconstrained optimization problems that encode constraints
via penalization. To solve these unconstrained optimization problems,
we use again a trust-region method based on BFGS-updates of the Hessian.
The augmented Lagrangian source code is
`here <https://github.com/trilinos/Trilinos/blob/master/packages/rol/src/step/ROL_AugmentedLagrangianStep.hpp>`_.

.. code-block:: none

    'Step':{'Type':'Augmented Lagrangian',
            'Augmented Lagrangian':{'Use Default Initial Penalty Parameter':true,
                                    'Initial Penalty Parameter':1.e1,
                                    # Multiplier update parameters
                                    'Use Scaled Augmented Lagrangian':false,
                                    'Penalty Parameter Reciprocal Lower Bound':0.1,
                                    'Penalty Parameter Growth Factor':1.e1,
                                    'Maximum Penalty Parameter':1.e8,
                                    # Optimality tolerance update
                                    'Optimality Tolerance Update Exponent':1,
                                    'Optimality Tolerance Decrease Exponent':1,
                                    'Initial Optimality Tolerance':1,
                                    # Feasibility tolerance update
                                    'Feasibility Tolerance Update Exponent':0.1,
                                    'easibility Tolerance Decrease Exponent':0.9,
                                    'Initial Feasibility Tolerance':1,
                                    # Subproblem information
                                    'Print Intermediate Optimization History':false,
                                    'Subproblem Iteration Limit':1000,
                                    'Subproblem Step Type':'Trust Region',
                                    # Scaling
                                    'Use Default Problem Scaling':true,
                                    'Objective Scaling':1.0,
                                    'Constraint Scaling':1.0,
                                   }
            'Trust Region':{'Initial Radius':-1, #determine initial radius with heuristics
                            'Maximum Radius':1.e8,
                            'Subproblem Solver':'Dogleg',
                            'Radius Growing Rate':2.5
                            'Step Acceptance Threshold':0.05,
                            'Radius Shrinking Threshold':0.05,
                            'Radius Growing Threshold':0.9,
                            'Radius Shrinking Rate (Negative rho)':0.0625,
                            'Radius Shrinking Rate (Positive rho)':0.25,
                            'Radius Growing Rate':2.5,
                            'Sufficient Decrease Parameter':1.e-4,
                            'Safeguard Size':100.0,
                           }
    'General':{'Print Verbosity':0, #set to any number >0 for increased verbosity
               'Secant':{'Type':'Limited-Memory BFGS', #BFGS-based Hessian-update in trust-region model
                         'Maximum Storage':10
                        }
              }

.. note::

    The augmented Lagrangian default constraint, gradient, and step
    tolerances for the outer iterations are set to :bash:`1.e-8`. The user
    can set different tolerances by specifying them in :bash:`Status Test`.
    The gradient and step tolerances for the internal iterations are
    set by the augmented Lagrangian algorithm itself (based on a number of parameters,
    including the gradient tolerance for the outer iteration, see lines 374-375 in the
    source code) and cannot be modified by the user.
