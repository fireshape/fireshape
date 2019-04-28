import firedrake as fd
import fireshape as fs
import ROL
from L2tracking_PDEconstraint import PoissonSolver
from L2tracking_objective import L2trackingObjective

# setup problem
mesh = fd.UnitSquareMesh(100, 100)
Q = fs.FeControlSpace(mesh)
inner = fs.ElasticityInnerProduct(Q)
q = fs.ControlVector(Q, inner)

# setup PDE constraint
mesh_m = Q.mesh_m
e = PoissonSolver(mesh_m)

# save state variable evolution in file u.pvd
e.solve()
out = fd.File("u.pvd")

# create PDEconstrained objective functional
J_ = L2trackingObjective(e, Q, cb=lambda: out.write(e.solution))
J = fs.ReducedObjective(J_, e)

# ROL parameters
params_dict = {
    'Step': {
        'Type': 'Line Search',
        'Line Search': {
            'Descent Method': {
                'Type': 'Quasi-Newton Step'
            }
        },
    },
    'General': {
        'Secant': {
            'Type': 'Limited-Memory BFGS',
            'Maximum Storage': 10
        }
    },
    'Status Test': {
        'Gradient Tolerance': 1e-4,
        'Step Tolerance': 1e-5,
        'Iteration Limit': 15
    }
}
params = ROL.ParameterList(params_dict, "Parameters")
problem = ROL.OptimizationProblem(J, q)
solver = ROL.OptimizationSolver(problem, params)
solver.solve()
