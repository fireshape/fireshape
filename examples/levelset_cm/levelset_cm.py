import firedrake as fd
import fireshape as fs
import ROL
from levelsetfunctional import LevelsetFunctional

# setup problem
mesh = fd.Mesh("/home/prem/src/github/fireshape/examples/levelset_cm/square_in_square.msh")

S = fd.FunctionSpace(mesh, "DG", 0)
I = fd.Function(S, name="indicator")
x = fd.SpatialCoordinate(mesh)

I.interpolate(fd.conditional(x[0] < 1, 
              fd.conditional(x[0] > 0,
              fd.conditional(x[1] > 0,
              fd.conditional(x[1] < 1, 1, 0), 0), 0), 0))

Q = fs.CmControlSpace(mesh, I)
# inner = fs.LaplaceInnerProduct(Q)
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
