import firedrake as fd
import fireshape as fs
import ROL
from L2tracking_PDEconstraint import PoissonSolver
from L2tracking_objective import L2trackingObjective
#import sys; sys.exit(0)

# create mesh on initial guess
mesh = fd.UnitSquareMesh(100,100)

# select inner product and control space
inner = fs.ElasticityInnerProduct()
Q = fs.FeControlSpace(mesh, inner)

# set PDE constraint on moving mesh
e = PoissonSolver(Q.mesh_m)

# define function that stores solution after each iteration
out = fd.File("u.pvd")
def cb(*args):
    out.write(e.solution)

# solve PDE constraint and store its solution before optimizing
e.solve()
cb()


J = L2trackingObjective(e, Q, cb=cb, scale=0.5)
Jr = fs.ReducedObjective(J, e)
q = fs.ControlVector(Q)
#g = q.clone()

# select optimization algorithm
params_dict = {
        'General': {
            'Secant': { 'Type': 'Limited-Memory BFGS', 'Maximum Storage': 10 } },
            'Step': {
                'Type': 'Augmented Lagrangian',
                'Line Search': {'Descent Method': {
                    'Type': 'Quasi-Newton Step'}
                    },
                'Augmented Lagrangian': {
                    'Subproblem Step Type': 'Line Search',
                    'Penalty Parameter Growth Factor': 2.,
                    'Print Intermediate Optimization History': True
                    }},
        'Status Test': {
            'Gradient Tolerance': 1e-15, 'Relative Gradient Tolerance': 1e-10,
            'Step Tolerance': 1e-16, 'Relative Step Tolerance': 1e-10,
            'Iteration Limit': 40}
        }

# assemble and solve optimization problem
params = ROL.ParameterList(params_dict, "Parameters")
problem = ROL.OptimizationProblem(Jr, q)
solver = ROL.OptimizationSolver(problem, params)
solver.solve()
