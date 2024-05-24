import firedrake as fd
import fireshape as fs
import fireshape.zoo as fsz
import petsc4py.PETSc as PETSc

# setup problem using multigrid controlspace: the physical mesh
# is twice as fine than the controlspace mesh
mesh = fd.UnitSquareMesh(3, 3)  # initial guess
Q = fs.FeMultiGridControlSpace(mesh, refinements=2, degree=2)
Q.assign_inner_product(fs.H1InnerProduct(Q))

# create objective functional, the optimum is
# a disc of radius 0.5 centered at (0.5, 0.5)
# set usecb=True to store mesh iterates in soln.pvd
x, y = fd.SpatialCoordinate(Q.mesh_m)
f = (x - 0.5)**2 + (y - 0.5)**2 - 0.5**2
J = fsz.LevelsetFunctional(Q, f, usecb=True)

# PETSc.TAO solver using the limited-memory
# variable-metric method. Call using
# python levelset_multigrid.py -tao_monitor
# to print updates in the terminal
solver = PETSc.TAO().create()
solver.setType("lmvm")
solver.setFromOptions()
solver.setSolution(Q.get_PETSc_zero_vec())
solver.setObjectiveGradient(J.objectiveGradient, None)
solver.setTolerances(gatol=1.0e-4, grtol=1.0e-4)
solver.solve()
