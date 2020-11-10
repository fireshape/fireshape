from firedrake import *
from fireshape import *
import ROL
from TimeTracking import TimeTracking

# setup problem
mesh = UnitSquareMesh(10, 10)
Q = FeControlSpace(mesh)
inner = LaplaceInnerProduct(Q, fixed_bids=[1, 2, 3, 4])
q = ControlVector(Q, inner)
J = TimeTracking(Q)

params_dict = {'Step': {'Type': 'Trust Region'},
               'General': {'Secant': {'Type': 'Limited-Memory BFGS',
                                      'Maximum Storage': 25}},
               'Status Test': {'Gradient Tolerance': 1e-6,
                               'Step Tolerance': 1e-8,
                               'Iteration Limit': 40}}

# assemble and solve ROL optimization problem
params = ROL.ParameterList(params_dict, "Parameters")
problem = ROL.OptimizationProblem(J, q)
solver = ROL.OptimizationSolver(problem, params)
solver.solve()
