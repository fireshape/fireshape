import firedrake as fd
import fireshape as fs
import fireshape.zoo as fsz
import petsc4py.PETSc as PETSc

# goal: optimise the mesh coordinates so that a function
# defined on the mesh approximates a target one
mesh = fd.UnitSquareMesh(30, 30)
Q = fs.FeControlSpace(mesh)
inner = fs.LaplaceInnerProduct(Q, fixed_bids=[1, 2, 3, 4])
Q.assign_inner_product(fs.LaplaceInnerProduct(Q))

# save shape evolution in file soln.pvd
V = fd.FunctionSpace(Q.mesh_m, "DG", 1)
sigma = fd.Function(V)  # function that moves with the mesh
x, y = fd.SpatialCoordinate(Q.mesh_m)
perturbation = 0.1*fd.sin(x*fd.pi)*(16*y**2*(1-y)**2)
sigma.interpolate(y*(1-y)*(fd.cos(2*fd.pi*x*(1+perturbation))))
f = fd.cos(2*fd.pi*x)*y*(1-y)  # target
out = fd.VTKFile("soln.pvd")
cb = lambda: out.write(sigma)
J = fsz.LevelsetFunctional(Q, (sigma - f)**2, cb=cb, quadrature_degree=2)

# PETSc.TAO solver using the limited-memory
# variable-metric method. Call using
# python levelset_density.py -tao_monitor
# to print updates in the terminal
solver = PETSc.TAO().create()
solver.setType("lmvm")
solver.setFromOptions()
solver.setSolution(Q.get_PETSc_zero_vec())
solver.setObjectiveGradient(J.objectiveGradient, None)
solver.setTolerances(gatol=1.0e-3, grtol=1.0e-3)
solver.solve()
