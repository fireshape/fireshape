import firedrake as fd
import fireshape as fs
import ROL
from L2tracking_PDEconstraint import PoissonSolver
from L2tracking_objective import L2trackingObjective

# Setup problem
mesh = fd.Mesh("mesh.msh")

bbox = [(-3., -1.), (-1., 1.)]
orders = [3, 3]
levels = [6, 6]
Q = fs.BsplineControlSpace(mesh, bbox, orders, levels,
                           boundary_regularities=[1, 1])
inner = fs.H1InnerProduct(Q)
q = fs.ControlVector(Q, inner)

# Setup PDE constraint
rt = 0.5
ct = (-1.9, 0.)
mesh_m = Q.mesh_m
e = PoissonSolver(mesh_m, rt, ct)

# Save state variable evolution in file u.pvd
e.solve()
out = fd.File("u.pvd")

# Create PDEconstrained objective functional
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
        }
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
