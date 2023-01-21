import firedrake as fd
import fireshape as fs
import ROL
from L2tracking_PDEconstraint import PoissonSolver
from L2tracking_objective import L2trackingObjective

# Setup problem
mesh = fd.Mesh("mesh.msh")

bbox = [(-3., -1.), (-1., 1.)]
primal_orders = [3, 3]
dual_orders = [3, 3]
levels = [6, 6]
norm_equiv = True
Q = fs.WaveletControlSpace(mesh, bbox, primal_orders, dual_orders, levels,
                           norm_equiv=norm_equiv, tol=1e-1)
inner = None if norm_equiv else fs.H1InnerProduct(Q)
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
J.update(q, None, 1)
g = q.clone()
J.gradient(g, q, None)

if norm_equiv:
    c = 1 / g.norm()
    J = c * J  # normalize gradient in first optimization step
else:
    from math import sqrt
    c = 1 / sqrt(g.vec_ro().dot(g.vec_ro()))

g.scale(c)
Q.visualize_control(g)

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
