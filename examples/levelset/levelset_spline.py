import firedrake as fd
import fireshape.zoo as fsz
import fireshape as fs
import ROL

n = 100
mesh = fd.UnitSquareMesh(n, n)

bbox = [(-3, 4), (-3, 4)]
orders = [2, 2]
levels = [3, 3]
Q = fs.BsplineControlSpace(mesh, bbox, orders, levels)
inner = fs.H1InnerProduct(Q, fixed_bids=[1, 2, 3, 4])
q = fs.ControlVector(Q, inner)

mesh_m = Q.mesh_m
(x, y) = fd.SpatialCoordinate(mesh_m)
f = (pow(x-0.5, 2))+pow(y-0.5, 2) - 2.

out = fd.File("domain.pvd")


J = fsz.LevelsetFunctional(f, Q, cb=lambda: out.write(mesh_m.coordinates))
J = 0.1 * J

params_dict = {
    'General': {
        'Secant': {'Type': 'Limited-Memory BFGS',
                   'Maximum Storage': 2}},
    'Step': {
        'Type': 'Line Search',
        'Line Search': {'Descent Method': {
            'Type': 'Quasi-Newton Step'}}},
    'Status Test': {
        'Gradient Tolerance': 1e-5,
        'Step Tolerance': 1e-6,
        'Iteration Limit': 50}}

params = ROL.ParameterList(params_dict, "Parameters")
problem = ROL.OptimizationProblem(J, q)
solver = ROL.OptimizationSolver(problem, params)
solver.solve()
