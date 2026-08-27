import firedrake as fd
import fireshape as fs
import ROL

from constraint_eigen import EigenObjective

# Set up the domain
n = 20
mesh = fd.UnitSquareMesh(n, n)
Q = fs.FeControlSpace(mesh)
inner = fs.LaplaceInnerProduct(Q)
q = fs.ControlVector(Q, inner)

# setup PDE

J = EigenObjective(Q)


params_dict = {'Step': {'Type': 'Trust Region'},
               'General': {'Secant': {'Type': 'Limited-Memory BFGS',
                                      'Maximum Storage': 25}},
               'Status Test': {'Gradient Tolerance': 1e-3,
                               'Step Tolerance': 1e-8,
                               'Iteration Limit': 20}}

params = ROL.ParameterList(params_dict, "Parameters")
problem = ROL.OptimizationProblem(J, q)
solver = ROL.OptimizationSolver(problem, params)
solver.solve()
