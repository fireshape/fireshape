import firedrake as fd
import fireshape as fs
import fireshape.zoo as fsz

import ROL

mesh_c = fd.Mesh("stoke_control.msh")
mesh_r = fd.Mesh("stoke_hole.msh")

# Q = fs.FeControlSpace(mesh)
# Q = fs.FeMultiGridControlSpace(mesh, refinements=1, degree=1)

S = fd.FunctionSpace(mesh_c, "DG", 0)
I = fd.Function(S, name="indicator")
x = fd.SpatialCoordinate(mesh_c)

I.interpolate(fd.conditional(x[0]**2 + x[1]**2 < 0.5**2, 1, 0))

Q = fs.CmControlSpace(mesh_c, mesh_r, I)
# this 1,2,3 represents the physical curve tag 
# have added 5 here as this is the complete outside bit
inner = fs.L2InnerProduct(Q, fixed_bids=[1, 2, 3])
mesh_m = Q.mesh_m

(x, y) = fd.SpatialCoordinate(mesh_m)
inflow_expr = fd.Constant((1.0, 0.0))
e = fsz.StokesSolver(mesh_m, inflow_bids=[1, 2],
                     inflow_expr=inflow_expr, noslip_bids=[4], direct=False)
e.solve()
out = fd.File("profiling.pvd")


def cb(*args):
    out.write(e.solution.split()[0])


cb()

Je = fsz.EnergyObjective(e, Q, cb=cb)
Jr = 1e-2 * fs.ReducedObjective(Je, e)
Js = fsz.MoYoSpectralConstraint(10., fd.Constant(0.7), Q)
# Jd = 1e-3 * fsz.DeformationRegularization(Q, l2_reg=1e-2, sym_grad_reg=1e0, skew_grad_reg=1e-2)
J = Jr + Js
q = fs.ControlVector(Q, inner)

d = q.clone()
# place 1s into d
# d.fun.assign(1) # this is irrelevant because of line 86
x = q.clone()
x.fun.assign(3)

J.update(x, None, -1)
# updates d to contain gradient (steepest direction) (just a random direction)
J.gradient(d, x, None)

# update mesh using control vector (x = 1) to take derivative around x
# J.update(x, None, -1)
# J.gradient(d, x, None)
J.checkGradient(x, d, 5, 1)
