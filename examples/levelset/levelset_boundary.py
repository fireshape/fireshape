import firedrake as fd
import fireshape as fs
import fireshape.zoo as fsz
import ROL

dim = 2
mesh = fs.DiskMesh(0.4)
Q = fs.FeMultiGridControlSpace(mesh, refinements=4, degree=2)
# Q = fs.FeControlSpace(mesh)
# inner = fs.SurfaceInnerProduct(Q)
inner = fs.ElasticityInnerProduct(Q)
extension = fs.ElasticityExtension(Q.V_r, direct_solve=True)


mesh_m = Q.mesh_m

if dim == 2:
    (x, y) = fd.SpatialCoordinate(mesh_m)
    f = (pow(x, 2))+pow(0.5*y, 2) - 1.
else:
    (x, y, z) = fd.SpatialCoordinate(mesh_m)
    f = (pow(x-0.5, 2))+pow(y-0.5, 2)+pow(z-0.5, 2) - 2.

q = fs.ControlVector(Q, inner, boundary_extension=extension)
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
        'Gradient Tolerance': 1e-5,
        'Step Tolerance': 1e-10,
        'Iteration Limit': 100}}

params = ROL.ParameterList(params_dict, "Parameters")
problem = ROL.OptimizationProblem(J, q)
solver = ROL.OptimizationSolver(problem, params)
solver.solve()
J.gradient(g, q, None)
J.checkGradient(q, g, 9, 1)
