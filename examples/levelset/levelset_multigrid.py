import firedrake as fd
import fireshape as fs
import fireshape.zoo as fsz
import ROL

n = 30
# mesh = fd.UnitSquareMesh(n, n)
mesh = fd.Mesh("UnitSquareCrossed.msh")
mesh = fd.MeshHierarchy(mesh, 1)[-1]

Q = fs.FeMultiGridControlSpace(mesh, refinements=3, order=2)
inner = fs.LaplaceInnerProduct(Q)
mesh_m = Q.mesh_m
V_m = fd.FunctionSpace(mesh_m, "CG", 1)
f_m = fd.Function(V_m)

(x, y) = fd.SpatialCoordinate(mesh_m)
f = (pow(x-0.5, 2))+pow(y-0.5, 2) - 2.
out = fd.File("domain.pvd")
J = fsz.LevelsetFunctional(f, Q, cb=lambda: out.write(mesh_m.coordinates))

q = fs.ControlVector(Q, inner)

params_dict = {
    'General': {
        'Secant': {'Type': 'Limited-Memory BFGS', 'Maximum Storage': 5}},
    'Step': {
        'Type': 'Line Search',
        'Line Search': {'Descent Method': {'Type': 'Quasi-Newton Step'}}},
    'Status Test': {
        'Gradient Tolerance': 1e-5,
        'Step Tolerance': 1e-6,
        'Iteration Limit': 40}}

params = ROL.ParameterList(params_dict, "Parameters")
problem = ROL.OptimizationProblem(J, q)
solver = ROL.OptimizationSolver(problem, params)
solver.solve()
