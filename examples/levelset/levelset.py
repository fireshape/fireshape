import firedrake as fd
import fireshape as fs
import fireshape.zoo as fsz
import ROL

dim = 2
if dim == 2:
    n = 100
    mesh = fd.UnitSquareMesh(n, n)
else:
    n = 30
    mesh = fd.UnitCubeMesh(n, n, n)

Q = fs.FeControlSpace(mesh)
inner = fs.LaplaceInnerProduct(Q)

mesh_m = Q.mesh_m

if dim == 2:
    (x, y) = fd.SpatialCoordinate(mesh_m)
    f = (pow(x-0.5, 2))+pow(y-0.5, 2) - 2.
else:
    (x, y, z) = fd.SpatialCoordinate(mesh_m)
    f = (pow(x-0.5, 2))+pow(y-0.5, 2)+pow(z-0.5, 2) - 2.

q = fs.ControlVector(Q, inner)
out = fd.File("domain.pvd")
J = fsz.LevelsetFunctional(f, Q, cb=lambda: out.write(mesh_m.coordinates))


params_dict = {
    'General': {
        'Secant': {'Type': 'Limited-Memory BFGS',
                   'Maximum Storage': 25}},
    'Step': {
        'Type': 'Line Search',
        'Line Search': {'Descent Method': {
            'Type': 'Quasi-Newton Step'}}},
    'Status Test': {
        'Gradient Tolerance': 1e-4,
        'Relative Gradient Tolerance': 1e-3,
        'Step Tolerance': 1e-10, 'Relative Step Tolerance': 1e-10,
        'Iteration Limit': 150}}

params = ROL.ParameterList(params_dict, "Parameters")
problem = ROL.OptimizationProblem(J, q)
solver = ROL.OptimizationSolver(problem, params)
solver.solve()
