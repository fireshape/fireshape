from firedrake import SpatialCoordinate, sqrt
import firedrake as fd
import fireshape as fs
import fireshape.zoo as fsz
import ROL
from params import get_params

mesh = fs.DiskMesh(0.025)
gradient_norms = []
out = fd.File("domain.pvd")
for i in range(20):
    Q = fs.FeScalarControlSpace(mesh, hessian_tangential=False, extension_tangential=True)
    inner = fs.SurfaceInnerProduct(Q)

    mesh_m = Q.mesh_m

    (x, y) = SpatialCoordinate(mesh_m)
    a = 0.8
    b = 2.0
    f = (sqrt((x - a)**2 + b * y**2) - 1) \
    * (sqrt((x + a)**2 + b * y**2) - 1) \
    * (sqrt(b * x**2 + (y - a)**2) - 1) \
    * (sqrt(b * x**2 + (y + a)**2) - 1) - 0.01

    q = fs.ControlVector(Q, inner)

    J = fsz.LevelsetFunctional(f, Q)
    g = q.clone()
    J.update(q, None, 1)
    J.gradient(g, q, None)
    g.scale(-0.3)
    J.update(g, None, 1)
    # J.checkGradient(q, g, 9, 1)
    # J.checkHessVec(q, g, 7, 1)

    def cb():
        J.gradient(g, None, None)
        gradient_norms.append(g.norm())
        out.write(mesh_m.coordinates)
    J.cb = cb
    if i==0:
        params = get_params("Quasi-Newton Method", 10)
        problem = ROL.OptimizationProblem(J, q)
        solver = ROL.OptimizationSolver(problem, params)
        solver.solve()
    params = get_params("Newton-Krylov", 1)
    problem = ROL.OptimizationProblem(J, q)
    solver = ROL.OptimizationSolver(problem, params)
    solver.solve()
    # J.gradient(g, q, None)
    # J.checkGradient(q, g, 9, 1)
    mesh = mesh_m

import matplotlib.pyplot as plt
plt.figure()
plt.semilogy([gradient_norms[i+1]/gradient_norms[i] for i in range(len(gradient_norms)-1)])
plt.show()
VV = fd.VectorFunctionSpace(mesh, "CG", 1)
V = fd.FunctionSpace(mesh, "CG", 1)

extension = fs.NormalExtension(VV, allow_tangential=False)

u = fd.Function(V)
out = fd.Function(VV)
u.interpolate(fd.Constant(1.0))
extension.extend(u, out)
fd.File("ext.pvd").write(out)
