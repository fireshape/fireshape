import fireshape as fs
from firedrake import *

mesh = UnitSquareMesh(30, 30)
# mesh = fs.DiskMesh(0.2)

VV = VectorFunctionSpace(mesh, "CG", 1)
V = FunctionSpace(mesh, "CG", 1)

extension = fs.NormalExtension(VV, allow_tangential=False)

u = Function(V)
out = Function(VV)

u.interpolate(Constant(1.0))

extension.extend(u, out)
File("u.pvd").write(u)
File("ext.pvd").write(out)
