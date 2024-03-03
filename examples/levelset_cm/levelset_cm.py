import firedrake as fd
from firedrake.__future__ import interpolate
import fireshape as fs
import ROL
from levelsetfunctional import LevelsetFunctional
from icecream import ic
# setup problem
mesh_c = fd.Mesh("square_in_square.msh")

S = fd.FunctionSpace(mesh_c, "DG", 0)
I = fd.Function(S, name="indicator")
x = fd.SpatialCoordinate(mesh_c)

I.interpolate(fd.conditional(x[0] < 1, fd.conditional(x[0] > 0, fd.conditional(x[1] > 0, fd.conditional(x[1] < 1, 1, 0), 0), 0), 0))

mesh_r = fd.UnitSquareMesh(30, 30)

Q = fs.CmControlSpace(mesh_c, mesh_r, I)
inner = fs.LaplaceInnerProduct(Q)

# inner = fs.H1InnerProduct(Q)
q = fs.ControlVector(Q, inner)

# save shape evolution in file domain.pvd
out = fd.File("laplace_inner_lower_tol.pvd")

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
        'Gradient Tolerance': 1e-5,
        'Step Tolerance': 1e-10,
        'Iteration Limit': 50,
    }
}

# Problem Optimising
params = ROL.ParameterList(params_dict, "Parameters")
problem = ROL.OptimizationProblem(J, q)
solver = ROL.OptimizationSolver(problem, params)
solver.solve()