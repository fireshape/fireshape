# Manual taylor test

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

# save shape evolution in file domain.pvd
out = fd.File("domain.pvd")

# create objective functional
J = LevelsetFunctional(Q)

x = q.clone()
x.fun.assign(1)

# # Gradient Checking
# creates 0 vector d of same size as q
d = q.clone()
# place 1s into d
d.fun.assign(1) # this is irrelevant because of line 86

# update mesh using control vector (x = 1) to take derivative around x
# updates d to contain gradient (steepest direction) (just a random direction)
J.gradient(d, x, None)

# # create a zero vector to store ∇J(x)
out = q.clone()

# # put ∇J(x) into out
J.update(x, None, -1)
J.derivative(out)

# # <∇J(x), d> 
print("Actual")
actual = fd.assemble(out.cofun(d.fun))
print(actual)

eps = [10**(-i) for i in range(4)]

print("\nNumerical")
for t in eps:
    # to do numerical approximations
    x2 = x.clone()
    x2.set(x)
    d2 = d.clone()
    d2.set(d)

    # x2 = x + dt
    d2.scale(t)
    x2.plus(d2)

    J.update(x2, None, -1)
    # a = J(x + td)
    a = J.value(None, None)

    J.update(x, None, -1)
    # b = J(x)
    b = J.value(None, None)


    print(t)
    print((a - b) / t)
    print(actual - ((a - b) / t))
    print()
