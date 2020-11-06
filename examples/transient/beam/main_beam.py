import firedrake as fd
import fireshape as fs
import fireshape.zoo as fsz
import ROL
from PDEconstraint_beam import CNBeamSolver

# setup problem
mesh = fd.RectangleMesh(40,4,1,0.1)
Q = fs.FeControlSpace(mesh)
inner = fs.LaplaceInnerProduct(Q, fixed_bids=[1])
q = fs.ControlVector(Q, inner)

e = CNBeamSolver(Q.mesh_m)

def cb():
    return


# create PDEconstrained objective functional
J = fs.TimeReducedObjective(Q,e,cb)

# Testing Derivatives
# J.update(q,None,0) # FIXME: This is needed to do one forward solve
# dJdct = J.Jred.derivative()
# temp = fd.Function(Q.V_r, name = "ShapePert")
# outgrad = fd.File("outgrad/test.pvd")
# temp.assign(dJdct)
# outgrad.write(temp)


# ROL parameters
params_dict = {
    'General': {'Print Verbosity': 0,  # set to 1 to understand output
                'Secant': {'Type': 'Limited-Memory BFGS',
                           'Maximum Storage': 10}},
    'Step': {'Type': 'Augmented Lagrangian',
             'Augmented Lagrangian':
             {'Subproblem Step Type': 'Trust Region',
              'Print Intermediate Optimization History': False,
              'Subproblem Iteration Limit': 10}},
    'Status Test': {'Gradient Tolerance': 1e-2,
                    'Step Tolerance': 1e-6,
                    'Constraint Tolerance': 1e-1,
                    'Iteration Limit': 3}}
params = ROL.ParameterList(params_dict, "Parameters")
problem = ROL.OptimizationProblem(J, q)
solver = ROL.OptimizationSolver(problem, params)
solver.solve()
