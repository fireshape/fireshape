import firedrake as fd
import fireshape as fs
import ROL
from utils import generate_mesh
from scattering_PDEconstraint import PMLSolver
from scattering_objective import FarFieldObjective


# Define initial guess
use_cached_mesh = False
a0 = b0 = 2.0
a1 = b1 = 2.25
R0 = 1.5
R1 = 1.9

if use_cached_mesh:
    mesh = fd.Mesh("mesh.msh")
else:
    obstacle = {
        "shape": "circle",
        "scale": 0.5,
        "nodes": 50
    }
    h0 = 0.1
    mesh = generate_mesh(a0, a1, b0, b1, R0, R1, obstacle, h0, name="mesh")

# Setup problem
bbox = [(-1, 1), (-1, 1)]
orders = [3, 3]
levels = [3, 3]
Q = fs.BsplineControlSpace(mesh, bbox, orders, levels,
                           boundary_regularities=[1, 1])
inner = fs.H1InnerProduct(Q)
q = fs.ControlVector(Q, inner)

# Setup PDE constraint
k = 1
d = (1., 0.)  # unit vector
mesh_m = Q.mesh_m
e = PMLSolver(mesh_m, k, d, a1, b1)

# Save state variable (real part) evolution in file u.pvd
e.solve()
out = fd.File("u.pvd")

# Create PDE-constrained objective functional
J = FarFieldObjective(e, R0, R1, Q, cb=lambda: out.write(e.solution.sub(0)))

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
