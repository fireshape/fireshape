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

mesh_r = fd.UnitSquareMesh(50, 50)

Q = fs.MultipleHelmholtzControlSpace(mesh_c, mesh_r, I, 25, 11, 0.05)
inner = fs.LaplaceInnerProduct(Q)

# inner = fs.H1InnerProduct(Q)
q = fs.ControlVector(Q, inner)

# save shape evolution in file domain.pvd
out = fd.File("alpha_scaling_005_higher_limit/moved.pvd")
out2 = fd.File("alpha_scaling_005_higher_limit/control.pvd")

control_copy = Q.mesh_c.coordinates.copy(deepcopy=True)

def cb():
    out.write(Q.mesh_m.coordinates)
    Q.mesh_c.coordinates.assign(Q.mesh_c.coordinates + Q.dphi)
    out2.write(Q.mesh_c.coordinates, I, Q.dphi)
    Q.mesh_c.coordinates.assign(control_copy)

# create objective functional
J = LevelsetFunctional(Q, cb=cb)

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
        'Gradient Tolerance': 1e-6,
        'Step Tolerance': 1e-10,
        'Iteration Limit': 1000,
    }
}

# Problem Optimising
params = ROL.ParameterList(params_dict, "Parameters")
problem = ROL.OptimizationProblem(J, q)
solver = ROL.OptimizationSolver(problem, params)
solver.solve()