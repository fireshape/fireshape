import firedrake as fd
import fireshape as fs
import ROL
import numpy as np
from utils import generate_mesh
from scattering_PDEconstraint import PMLSolver
from scattering_objective import FarFieldObjective


# Define initial guess
use_cached_mesh = False
a0 = b0 = 2.0
a1 = b1 = 2.25
R0 = 1.5
R1 = 1.9
layer = [(a0, a1), (b0, b1)]

if use_cached_mesh:
    mesh = fd.Mesh("mesh.msh", name="mesh")
else:
    obstacle = {
        "shape": "circle",
        "scale": 0.4,
    }
    refine = 1
    mesh = generate_mesh(obstacle, layer, R0, R1, refine, name="mesh")

# Setup problem
bbox = [(-1, 1), (-1, 1)]
primal_orders = [3, 3]
dual_orders = [3, 3]
levels = [5, 5]
norm_equiv = True
tol = 0.1
Q = fs.WaveletControlSpace(mesh, bbox, primal_orders, dual_orders, levels,
                           tol=tol)
inner = fs.H2InnerProduct(Q)
Q.assign_inner_product(inner)
if norm_equiv:
    q = fs.ControlVector(Q, None)
else:
    q = fs.ControlVector(Q, inner)

# Setup PDE constraint
k = 5
n_wave = 2
theta = 2 * np.pi / n_wave * np.arange(n_wave)
dirs = [(np.cos(t), np.sin(t)) for t in theta]
mesh_m = Q.mesh_m
e = PMLSolver(mesh_m, k, dirs, a1, b1)

# Save state variable (real part) evolution in file u.pvd
e.solve()
out = fd.File("u.pvd")

# Create PDE-constrained objective functional
J = FarFieldObjective(e, R0, R1, layer, Q,
                      cb=lambda: out.write(e.solutions[0]))
J.update(q, None, 1)
g = q.clone()
J.gradient(g, q, None)

if norm_equiv:
    c = 1 / g.norm()
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
        },
    },
    'General': {
        'Secant': {
            'Type': 'Limited-Memory BFGS',
            'Maximum Storage': 10
        }
    },
    'Status Test': {
        'Gradient Tolerance': 1e-3,
        'Step Tolerance': 1e-4,
        'Iteration Limit': 30
    }
}

params = ROL.ParameterList(params_dict, "Parameters")
problem = ROL.OptimizationProblem(J, q)
solver = ROL.OptimizationSolver(problem, params)
solver.solve()
