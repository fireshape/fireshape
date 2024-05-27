import firedrake as fd
import fireshape as fs
import fireshape.zoo as fsz
import petsc4py.PETSc as PETSc

# create initial domain
mesh = fd.UnitDiskMesh(refinement_level=2)

# create decoupled FE-controlspace on mesh_c using CG2 FE
mesh_c = fd.RectangleMesh(10, 10, 1.5, 1.5, -.5, -1.5)
fd.VTKFile("domain_control.pvd").write(mesh_c.coordinates)
# use also CG2 FE on mesh_r to represent self.T
Q = fs.FeControlSpace(mesh, add_to_degree_r=1, mesh_c=mesh_c, degree_c=2)
Q.assign_inner_product(fs.H1InnerProduct(Q, fixed_bids=[1, 2, 3, 4]))

# define objective so that optimal shape is
# a disc with center (0,0) and radius 1.2
x, y = fd.SpatialCoordinate(Q.mesh_m)
f = (pow(x, 2))+pow(y, 2) - 1.2**2
J = fsz.LevelsetFunctional(Q, f, usecb=True)

# PETSc.TAO solver using the limited-memory
# variable-metric method. Call using
# python levelset_fedecoupled.py -tao_monitor
# to print updates in the terminal
solver = PETSc.TAO().create()
solver.setType("lmvm")
solver.setFromOptions()
solver.setSolution(Q.get_PETSc_zero_vec())
solver.setObjectiveGradient(J.objectiveGradient, None)
solver.setTolerances(gatol=1.0e-2, grtol=1.0e-2)
solver.solve()
