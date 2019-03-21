from firedrake import SpatialCoordinate, sqrt # noqa
import firedrake as fd
import fireshape as fs
import fireshape.zoo as fsz
import ROL
from params import get_params
import matplotlib.pyplot as plt
import argparse

"""
call with:
    python3 stokes_newton.py --bfgs_iter 6 --newton_iter 10 #######
"""

parser = argparse.ArgumentParser(add_help=False)
parser.add_argument("--bfgs_iter", type=int, default=6)
parser.add_argument("--newton_iter", type=int, default=20)
args, _ = parser.parse_known_args()

mesh = fs.DiskMesh(0.1, radius=1.5)
gradient_norms = []
out = fd.File("domain.pvd")

Q = fs.FeScalarControlSpace(mesh, hessian_tangential=True,
                            extension_tangential=True)
inner = fs.SurfaceInnerProduct(Q)

mesh_m = Q.mesh_m

(x, y) = SpatialCoordinate(mesh_m)
# a = 0.8
# b = 2.0
# f = (sqrt((x - a)**2 + b * y**2) - 1) \
# * (sqrt((x + a)**2 + b * y**2) - 1) \
# * (sqrt(b * x**2 + (y - a)**2) - 1) \
# * (sqrt(b * x**2 + (y + a)**2) - 1) - 0.01

f = (x**2 + y**2 - 1) * (x**2 + (y-1)**2 - 0.5) - 1e-2

q = fs.ControlVector(Q, inner)

J = fsz.LevelsetFunctional(f, Q)
g = q.clone()
J.update(q, None, 1)
J.gradient(g, q, None)
g.scale(-0.3)
J.update(g, None, 1)
J.checkGradient(q, g, 9, 1)
J.checkHessVec(q, g, 7, 1)


def cb():
    J.gradient(g, None, None)
    gradient_norms.append(g.norm())
    out.write(mesh_m.coordinates)


J.cb = cb

if args.bfgs_iter > 0:
    params = get_params("Quasi-Newton Method", args.bfgs_iter)
    problem = ROL.OptimizationProblem(J, q)
    solver = ROL.OptimizationSolver(problem, params)
    solver.solve()

if args.newton_iter > 0:
    params = get_params("Newton-Krylov", args.newton_iter, ksp_type="GMRES")
    problem = ROL.OptimizationProblem(J, q)
    solver = ROL.OptimizationSolver(problem, params)
    solver.solve()


plt.figure()
plt.semilogy(gradient_norms)
plt.savefig("convergence.pdf")

mesh = mesh_m
# VV = fd.VectorFunctionSpace(mesh, "CG", 1)
# V = fd.FunctionSpace(mesh, "CG", 1)

# extension = fs.NormalExtension(VV, allow_tangential=False)

# u = fd.Function(V)
# out = fd.Function(VV)
# u.interpolate(fd.Constant(1.0))
# extension.extend(u, out)
# fd.File("ext.pvd").write(out)
