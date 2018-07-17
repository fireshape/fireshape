from firedrake import SpatialCoordinate, sqrt
import firedrake as fd
import fireshape as fs
import fireshape.zoo as fsz
import ROL

mesh = fs.DiskMesh(0.1)
Q = fs.FeScalarControlSpace(mesh, allow_tangential=True)
inner = fs.SurfaceInnerProduct(Q)

mesh_m = Q.mesh_m

(x, y) = SpatialCoordinate(mesh_m)
a = 0.8
b = 2.0
f = (sqrt((x - a)**2 + b * y**2) - 1) \
* (sqrt((x + a)**2 + b * y**2) - 1) \
* (sqrt(b * x**2 + (y - a)**2) - 1) \
* (sqrt(b * x**2 + (y + a)**2) - 1) - 0.002



q = fs.ControlVector(Q, inner)
out = fd.File("domain.pvd")
J = fsz.LevelsetFunctional(f, Q, cb=lambda: out.write(mesh_m.coordinates))
J.cb()
g = q.clone()
J.update(q, None, 1)
J.gradient(g, q, None)
g.scale(-0.3)
J.update(g, None, 1)
J.checkGradient(q, g, 9, 1)


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
        'Step Tolerance': 1e-10,
        'Iteration Limit': 100}}

params = ROL.ParameterList(params_dict, "Parameters")
problem = ROL.OptimizationProblem(J, q)
solver = ROL.OptimizationSolver(problem, params)
solver.solve()
J.gradient(g, q, None)
J.checkGradient(q, g, 9, 1)
