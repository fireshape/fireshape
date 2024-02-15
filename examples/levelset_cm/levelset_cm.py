import firedrake as fd
import fireshape as fs
import ROL
from levelsetfunctional import LevelsetFunctional
# setup problem
mesh = fd.Mesh("square_in_square.msh")

S = fd.FunctionSpace(mesh, "DG", 0)
I = fd.Function(S, name="indicator")
x = fd.SpatialCoordinate(mesh)

I.interpolate(fd.conditional(x[0] < 1, fd.conditional(x[0] > 0, fd.conditional(x[1] > 0, fd.conditional(x[1] < 1, 1, 0), 0), 0), 0))

mesh_m = fd.UnitSquareMesh(30, 30)

Q = fs.CmControlSpace(mesh, mesh_m, I)
# inner = fs.LaplaceInnerProduct(Q)
inner = fs.H1InnerProduct(Q)
q = fs.ControlVector(Q, inner)
# watchpoints.watch(q.fun)

# save shape evolution in file domain.pvd
out = fd.File("standard.pvd")

# TODO: Remove below indicator and fix how to plot in general
# Just creating this so I can actually plot it onto the output to see how shape changes
# S_2 = fd.FunctionSpace(Q.mesh_m, "DG", 0)
# I_2 = fd.Function(S_2, name="indicator")
# x_m, y_m = fd.SpatialCoordinate(Q.mesh_m)
# I_2.interpolate(fd.conditional(x_m < 1, fd.conditional(x_m > 0, fd.conditional(y_m > 0, fd.conditional(y_m < 1, 1, 0), 0), 0), 0))

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
        'Iteration Limit': 30,
    }
}

# params = ROL.ParameterList(params_dict, "Parameters")
# problem = ROL.OptimizationProblem(J, q)
# solver = ROL.OptimizationSolver(problem, params)
# solver.solve()

g = q.clone()
J.gradient(g, q, None)
q.plus(g)
J.update(q, None, 1)

J.gradient(g, q, None)
res = J.checkGradient(q, g, 5, 1)
print(res)
errors = [l[-1] for l in res]
print(errors)
# (errors[-1] < 0.11 * errors[-2])
# q.scale(0)
