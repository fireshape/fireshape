import firedrake as fd
import fireshape as fs
import fireshape.zoo as fsz
import ROL

mesh = fd.UnitDiskMesh(refinement_level=3)
Q = fs.FeControlSpace(mesh)
# Q = fs.FeMultiGridControlSpace(mesh, refinements=4, degree=2)
# inner = fs.SurfaceInnerProduct(Q)
# extension = fs.ElasticityExtension(Q.get_space_for_inner()[0],
#                                    direct_solve=True)
inner = fs.ElasticityInnerProduct(Q)
extension = None


mesh_m = Q.mesh_m

(x, y) = fd.SpatialCoordinate(mesh_m)
f = (pow(x, 2))+pow(0.5*y, 2) - 1.

q = fs.ControlVector(Q, inner, boundary_extension=extension)
out = fd.VTKFile("domain.pvd")
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
        'Gradient Tolerance': 1e-3,
        'Step Tolerance': 1e-5,
        'Iteration Limit': 50}}

params = ROL.ParameterList(params_dict, "Parameters")
problem = ROL.OptimizationProblem(J, q)
solver = ROL.OptimizationSolver(problem, params)
solver.solve()
J.gradient(g, q, None)
J.checkGradient(q, g, 9, 1)
