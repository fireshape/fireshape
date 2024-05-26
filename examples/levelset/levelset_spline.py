import firedrake as fd
import fireshape as fs
import fireshape.zoo as fsz
import petsc4py.PETSc as PETSc

# setup problem using BsplineControlSpace:
mesh = fd.UnitSquareMesh(10, 10)
bbox = [(-0.01, 1.01), (-0.01, 1.01)]
orders = [3, 3]
levels = [4, 4]
Q = fs.BsplineControlSpace(mesh, bbox, orders, levels,
                           boundary_regularities=[0, 0])
Q.assign_inner_product(fs.H1InnerProduct(Q))

# create objective functional, the optimum is
# a disc of radius 0.5 centered at (0.5, 0.5)
# set usecb=True to store mesh iterates in soln.pvd
x, y = fd.SpatialCoordinate(Q.mesh_m)
f = 0.1*((x - 0.5)**2 + (y - 0.5)**2 - 0.5**2)
J = fsz.LevelsetFunctional(Q, f, usecb=True)

# PETSc.TAO solver using the limited-memory
# variable-metric method. Call using
# python levelset_spline.py -tao_monitor
# to print updates in the terminal
solver = PETSc.TAO().create()
solver.setType("lmvm")
solver.setFromOptions()
solver.setSolution(Q.get_PETSc_zero_vec())
solver.setObjectiveGradient(J.objectiveGradient, None)
solver.setTolerances(gatol=1.0e-4, grtol=1.0e-4)
solver.solve()
