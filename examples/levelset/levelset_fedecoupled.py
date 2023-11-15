import firedrake as fd
import fireshape as fs
import fireshape.zoo as fsz
import ROL

mesh = fd.UnitDiskMesh(refinement_level=2)
#mesh_c = fd.UnitSquareMesh(5,5)
mesh_c = fd.RectangleMesh(10, 10, 1.5, 1.5, -.5, -1.5)
fd.File("domain_control.pvd").write(mesh_c.coordinates)

# create decouple FE-controlspace
Q = fs.FeControlSpace(mesh, add_to_degree_r=1, mesh_c=mesh_c, degree_c=2)
inner = fs.H1InnerProduct(Q, fixed_bids=[1, 2, 3, 4])

# define objective
mesh_m = Q.mesh_m
(x, y) = fd.SpatialCoordinate(mesh_m)
#f = (pow(x-0.5, 2))+pow(y-0.5, 2) - 2.
f = (pow(x-0., 2))+pow(y-0., 2) - 1.2**2
out = fd.File("domain.pvd")
J = fsz.LevelsetFunctional(f, Q, cb=lambda: out.write(mesh_m.coordinates))

# optimize
q = fs.ControlVector(Q, inner)
params_dict = {
    'General': {
        'Secant': {'Type': 'Limited-Memory BFGS', 'Maximum Storage': 5}},
    'Step': {
        'Type': 'Line Search',
        'Line Search': {'Descent Method': {'Type': 'Quasi-Newton Step'}}},
    'Status Test': {
        'Gradient Tolerance': 1e-1,
        'Step Tolerance': 1e-2,
        'Iteration Limit': 10}}

params = ROL.ParameterList(params_dict, "Parameters")
problem = ROL.OptimizationProblem(J, q)
solver = ROL.OptimizationSolver(problem, params)
solver.solve()
