import firedrake as fd
import fireshape as fs
import ROL
from levelsetfunctional import LevelsetFunctional

# setup problem
mesh = fd.UnitSquareMesh(2, 2, diagonal="crossed")
mh = fd.MeshHierarchy(mesh, 4)
# Q = fs.FeControlSpace(mesh)
Q = fs.FeMultiGridControlSpace(mh, order=2)
inner = fs.H1InnerProduct(Q)
q = fs.ControlVector(Q, inner)

# save shape evolution in file domain.pvd
out = fd.File("domain.pvd")

# create objective functional
J = LevelsetFunctional(Q, cb=lambda: out.write(Q.mesh_m.coordinates))

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
            'Maximum Storage': 25
        }
    },
    'Status Test': {
        'Gradient Tolerance': 1e-4,
        'Step Tolerance': 1e-10,
        'Iteration Limit': 30
    }
}

params = ROL.ParameterList(params_dict, "Parameters")
problem = ROL.OptimizationProblem(J, q)
solver = ROL.OptimizationSolver(problem, params)
solver.solve()
