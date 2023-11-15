import firedrake as fd
import fireshape as fs
import fireshape.zoo as fsz
import ROL

mesh = fd.UnitSquareMesh(3, 3)

# create multigrid controlspace: the physical mesh is twice
# as fine than the controlspace mesh
Q = fs.FeMultiGridControlSpace(mesh, refinements=2, degree=2)
inner = fs.H1InnerProduct(Q)

# define objective
mesh_m = Q.mesh_m
(x, y) = fd.SpatialCoordinate(mesh_m)
f = (pow(x-0.5, 2))+pow(y-0.5, 2) - 2.
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
        'Gradient Tolerance': 1e-3,
        'Step Tolerance': 1e-6,
        'Iteration Limit': 40}}

params = ROL.ParameterList(params_dict, "Parameters")
problem = ROL.OptimizationProblem(J, q)
solver = ROL.OptimizationSolver(problem, params)
solver.solve()
