from firedrake import SpatialCoordinate, sqrt
import firedrake as fd
import fireshape as fs
import fireshape.zoo as fsz
import ROL
from params import get_params
from volume_penalty import DomainVolumePenalty

mesh = fd.Mesh("Sphere2D.msh")
gradient_norms = []
out = fd.File("domain.pvd")
for i in range(2):
    Q = fs.FeScalarControlSpace(mesh, hessian_tangential=True, extension_tangential=True)
    inner = fs.SurfaceInnerProduct(Q, free_bids=[4])
    # Q = fs.FeControlSpace(mesh)
    # inner = fs.ElasticityInnerProduct(Q, fixed_bids=[1, 2, 3])

    mesh_m = Q.mesh_m
    q = fs.ControlVector(Q, inner)

    inflow_expr = fd.Constant((1.0, 0.0))
    e = fsz.StokesSolver(mesh_m, inflow_bids=[1, 2],
                         inflow_expr=inflow_expr, noslip_bids=[4])
    e.solve()
    Je = fsz.EnergyObjective(e, Q)
    Jr = fs.ReducedObjective(Je, e)
    # vol = fsz.LevelsetFunctional(fd.Constant(1.0), Q)
    vol = DomainVolumePenalty(Q, target_volume=47.21586287736358)
    # J = 1e-2 * Jr + 1 * vol
    J =  Jr + 1 * vol
    g = q.clone()
    J.update(q, None, 1)
    J.gradient(g, q, None)
    g.scale(-0.3)
    J.update(g, None, 1)
    # J.checkGradient(q, g, 9, 1)
    Jr.checkHessVec(q, g, 7, 1)

    def cb():
        J.gradient(g, None, None)
        gradient_norms.append(g.norm())
        out.write(mesh_m.coordinates)
    Jr.cb = cb
    if i==0:
        params = get_params("Quasi-Newton Method", 1)
        problem = ROL.OptimizationProblem(J, q)
        solver = ROL.OptimizationSolver(problem, params)
        solver.solve()
        # import sys; sys.exit(1)
    else:
        params = get_params("Newton-Krylov", 12, ksp_type="GMRES")
        problem = ROL.OptimizationProblem(J, q)
        solver = ROL.OptimizationSolver(problem, params)
        solver.solve()
    # J.gradient(g, q, None)
    # J.checkGradient(q, g, 9, 1)
    mesh = mesh_m

import matplotlib.pyplot as plt
plt.figure()
plt.semilogy([gradient_norms[i+1]/gradient_norms[i] for i in range(len(gradient_norms)-1)])
# plt.semilogy(gradient_norms)
plt.show()
VV = fd.VectorFunctionSpace(mesh, "CG", 1)
V = fd.FunctionSpace(mesh, "CG", 1)

extension = fs.NormalExtension(VV, allow_tangential=True)

u = fd.Function(V)
out = fd.Function(VV)
u.interpolate(fd.Constant(1.0))
extension.extend(u, out)
fd.File("ext2.pvd").write(out)
from IPython import embed; embed()
