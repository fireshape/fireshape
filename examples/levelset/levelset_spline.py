import firedrake as fd
import fireshape.zoo as fsz
import fireshape as fs
import ROL

mesh = fs.DiskMesh(0.1)

bbox = [(-1.01, 1.01), (-1.01, 1.01)]
orders = [3, 3]
levels = [4, 4]
Q = fs.BsplineControlSpace(mesh, bbox, orders, levels, boundary_regularities=[0, 0])
inner = fs.ElasticityInnerProduct(Q)
q = fs.ControlVector(Q, inner)

mesh_m = Q.mesh_m
(x, y) = fd.SpatialCoordinate(mesh_m)

f = (pow(x, 2))+pow(2*y, 2) - 1
outdef = fd.File("deformation.pvd")
V, I = Q.get_space_for_inner()
T = fd.Function(V)
def cb():
    out.write(mesh_m.coordinates)
    Q.visualize_control(q, T)
    outdef.write(T)

out = fd.File("domain.pvd")


J = fsz.LevelsetFunctional(f, Q, cb=cb)
J = 0.1 * J

g = q.clone()
J.gradient(g, q, None)
J.checkGradient(q, g, 4, 1)


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
