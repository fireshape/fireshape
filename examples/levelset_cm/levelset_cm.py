import firedrake as fd
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
# inner = fs.LaplaceInnerProduct(Q)

inner = fs.H1InnerProduct(Q)
q = fs.ControlVector(Q, inner)

# watchpoints.watch(q.fun)

# save shape evolution in file domain.pvd
out = fd.File("domain.pvd")

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

# Problem Optimising
# params = ROL.ParameterList(params_dict, "Parameters")
# problem = ROL.OptimizationProblem(J, q)
# solver = ROL.OptimizationSolver(problem, params)
# solver.solve()





# for e in eps:
x = q.clone()
x.fun.assign(1)

# # Gradient Checking
# creates 0 vector d of same size as q
d = q.clone()
# place 1s into d
d.fun.assign(1) # this is irrelevant because of line 86


# update mesh using control vector (x = 1) to take derivative around x
J.update(x, None, -1)

# updates d to contain gradient (steepest direction) (just a random direction)
J.gradient(d, x, None)

# create a zero vector to store ∇J(x)
out = q.clone()

# put ∇J(x) into out
J.derivative(out)

# <∇J(x), d> 
print("Actual")
print(fd.assemble(out.cofun(d.fun)))

eps = [10**(-i) for i in range(1, 4)]

print("\nNumerical")
for t in eps:
    # to do numerical approximations
    x2 = x.clone()
    x2.set(x)
    d2 = d.clone()
    d2.set(d)

    d2.scale(t)
    x2.plus(d2)
    J.update(x2, None, -1)
    # J(x + td)
    a = J.value(None, None)

    J.update(x, None, -1)
    # J(x)
    b = J.value(None, None)

    print((a - b) / t)


# How fireshape runs taylor test using ROL
J.update(x, None, -1)
J.gradient(d, x, None)
J.checkGradient(x, d, 4, 1)

# Actual
# 0.0012737007636444402
#
# Numerical
# -5.145805128373773 t = 0.1
# -70.17211896991135 t = 0.01
# -827.4544674147593 t = 0.001
#
# ROL
#            Step size           grad'*dir           FD approx           abs error
#            ---------           ---------           ---------           ---------
#    1.00000000000e+00   2.95976979305e-02   8.87010684158e-01   8.57412986227e-01
#    1.00000000000e-01   2.95976979305e-02   1.80265258045e+01   1.79969281066e+01
#    1.00000000000e-02   2.95976979305e-02   2.75013915917e+02   2.74984318219e+02
#    1.00000000000e-03   2.95976979305e-02   3.72542228599e+03   3.72539268829e+03


# Actual
# 0.0012737007636444402
#
# Numerical
# -0.5161235034503135 t = 1
# -7.028500477740027  t = 0.1
# -82.8238660000856   t = 0.01
# -914.4116158400024  t = 0.001
#
# ROL
#            Step size           grad'*dir           FD approx           abs error
#            ---------           ---------           ---------           ---------
#    1.00000000000e+00   3.81877486928e-02   4.94257487513e-01   4.56069738820e-01
#    1.00000000000e-01   3.81877486928e-02   5.10045118926e+00   5.06226344056e+00
#    1.00000000000e-02   3.81877486928e-02   5.24081820179e+01   5.23699942692e+01
#    1.00000000000e-03   3.81877486928e-02   5.32678693286e+02   5.32640505537e+02