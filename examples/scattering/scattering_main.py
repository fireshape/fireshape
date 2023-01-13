import firedrake as fd
import fireshape as fs
import ROL
import numpy as np
from utils import generate_mesh, plot_mesh
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
    mesh = generate_mesh(obstacle, layer, R0, R1, 1, name="mesh")

# Setup problem
bbox = [(-1, 1), (-1, 1)]
orders = [3, 3]
levels = [5, 5]
Q = fs.BsplineControlSpace(mesh, bbox, orders, levels,
                           boundary_regularities=[1, 1])
inner = fs.H1InnerProduct(Q)
q = fs.ControlVector(Q, inner)
plot_mesh(mesh, bbox, "mesh")

# Setup PDE constraint
k = 5
n_wave = 4
theta = 2 * np.pi / n_wave * np.arange(n_wave)
dirs = [(np.cos(t), np.sin(t)) for t in theta]
mesh_m = Q.mesh_m
e = PMLSolver(mesh_m, k, dirs, a1, b1)

# Save state variable (real part) evolution in file u.pvd
e.solve()
out = fd.File("u.pvd")

# Create PDE-constrained objective functional
J = FarFieldObjective(e, R0, R1, layer, Q,
                      cb=lambda: out.write(e.solutions[0].sub(0)))
J.update(q, None, 1)
g = q.clone()
J.gradient(g, q, None)
J.checkGradient(q, g, 7, 1)

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
        'Gradient Tolerance': 1e-2,
        'Step Tolerance': 1e-3,
        'Iteration Limit': 15
    }
}
params = ROL.ParameterList(params_dict, "Parameters")
problem = ROL.OptimizationProblem(J, q)
solver = ROL.OptimizationSolver(problem, params)
solver.solve()
