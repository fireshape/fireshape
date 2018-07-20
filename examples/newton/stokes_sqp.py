from firedrake import SpatialCoordinate, sqrt
import firedrake as fd
import fireshape as fs
import fireshape.zoo as fsz
import ROL
from params import get_params
from volume_penalty import DomainVolumePenalty
from sqp import sqp, SQP

mesh = fd.Mesh("Sphere2D.msh")
gradient_norms = []
out = fd.File("u.pvd")
Q = fs.FeScalarControlSpace(mesh, hessian_tangential=True, extension_tangential=True)
inner = fs.SurfaceInnerProduct(Q, free_bids=[4])
mesh_m = Q.mesh_m
q = fs.ControlVector(Q, inner)

inflow_expr = fd.Constant((1.0, 0.0))
e = fsz.StokesSolver(mesh_m, inflow_bids=[1, 2],
                     inflow_expr=inflow_expr, noslip_bids=[4])
e.solve()
Je = fsz.EnergyObjective(e, Q)
Jr = fs.ReducedObjective(Je, e)
vol = DomainVolumePenalty(Q, target_volume=47.21586287736358)
J = Jr + 1 * vol
g = q.clone()
J.update(q, None, 1)
J.gradient(g, q, None)
g.scale(-0.3)
J.update(g, None, 1)
# J.checkGradient(q, g, 9, 1)
# J.checkHessVec(q, g, 7, 1)

sqp = SQP(q, Je, e, vol)
def cb():
    # J.gradient(g, None, None)
    # gradient_norms.append(g.norm())
    out.write(e.solution.split()[0])
    # sqp(q, Je, e, vol)
    sqp.get_dL().norm()
Jr.cb = cb
params = get_params("Quasi-Newton Method", 5)
problem = ROL.OptimizationProblem(J, q)
solver = ROL.OptimizationSolver(problem, params)
solver.solve()
# sqp.taylor_test()

for i in range(5):
    sqp.step()
    cb()
# params = get_params("Newton-Krylov", 20)
# problem = ROL.OptimizationProblem(J, q)
# solver = ROL.OptimizationSolver(problem, params)
# solver.solve()

# sqp(q, Je, e, vol)
# e.solve()
# sqp(q, Je, e)
# cb()
# e.solve()
