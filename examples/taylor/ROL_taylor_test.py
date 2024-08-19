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
inner = fs.H1InnerProduct(Q)
q = fs.ControlVector(Q, inner)

# create objective functional
J = LevelsetFunctional(Q, cb=lambda: out.write(Q.mesh_m.coordinates))

# # Gradient Checking
# creates 0 vector d of same size as q
for i in range(10):
    d = q.clone()
    # place 1s into d
    # d.fun.assign(1) # this is irrelevant because of line 86
    x = q.clone()
    x.fun.assign(i)

    # updates d to contain gradient (steepest direction) (just a random direction)
    J.gradient(d, x, None)

    # update mesh using control vector (x = 1) to take derivative around x
    J.update(x, None, -1)
    J.gradient(d, x, None)
    J.checkGradient(x, d, 4, 1)

# Using same notation as from ROL docs
# https://docs.trilinos.org/dev/packages/rol/doc/html/classROL_1_1Objective.html