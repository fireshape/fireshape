import firedrake as fd
import fireshape as fs
import fireshape.zoo as fsz
import petsc4py.PETSc as PETSc

# goal: update the mesh coordinates of a periodic mesh
# so that a function defined on the mesh becomes a target one
mesh = fd.PeriodicUnitSquareMesh(15, 15)
Q = fs.FeControlSpace(mesh)
# in this example, LaplaceInnerProduct and ElasticityInnerProduct
# work better because they exclude translations (and rotations)
Q.assign_inner_product(fs.LaplaceInnerProduct(Q))
#Q.assign_inner_product(fs.ElasticityInnerProduct(Q))

# save shape evolution in file soln.pvd
V = fd.FunctionSpace(Q.mesh_m, "DG", 1)
sigma = fd.Function(V)  # function that moves with the mesh
x, y = fd.SpatialCoordinate(Q.mesh_m)
g = fd.sin(y*fd.pi)  # truncate at bdry
f = fd.cos(2*fd.pi*x)*g  # target function
perturbation = 0.1*fd.sin(x*fd.pi)*g**2  # perturb sigma
sigma.interpolate(g*fd.cos(2*fd.pi*x*(1+perturbation)))
out = fd.VTKFile("soln.pvd")
cb = lambda: out.write(sigma)
J = fsz.LevelsetFunctional(Q, (sigma - f)**2, cb=cb, quadrature_degree=2)

# PETSc.TAO solver using the limited-memory
# variable-metric method. Call using
# python levelset_periodic.py -tao_monitor
# to print updates in the terminal
solver = PETSc.TAO().create()
solver.setType("lmvm")
solver.setFromOptions()
solver.setSolution(Q.get_PETSc_zero_vec())
solver.setObjectiveGradient(J.objectiveGradient, None)
solver.setTolerances(gatol=1.0e-3, grtol=1.0e-3)
solver.solve()
