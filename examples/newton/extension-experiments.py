from firedrake import *
import fireshape as fs
Lx = 1
Ly = 1
Nx = 3
Ny = int(Nx * Ly/Lx)
# mesh = RectangleMesh(Nx, Ny, Lx, Ly, diagonal="crossed")
# mesh = MeshHierarchy(UnitTriangleMesh(), 2)[-1]
mesh = fs.DiskMesh(0.1, shiftx=0.5, shifty=0.5)
# mesh = fs.SphereMesh(0.5)

n = FacetNormal(mesh)
t = as_vector([n[1], -n[0]])
p = 1
V = VectorFunctionSpace(mesh, "CG", p)

X = SpatialCoordinate(mesh)
x, y = X[0], X[1]
g = Constant(0.1)
# g = y**2*x**2
g = Function(FunctionSpace(mesh, "CG", p)).interpolate(g)

zeroTangent = False
if zeroTangent:
    Q = VectorFunctionSpace(mesh, "CG", p); bcmethod = "topological"
else:
    Q = FunctionSpace(mesh, "HDivT", p); bcmethod = "topological"
    # Q = FunctionSpace(mesh, "CR", 1); bcmethod = "topological"
    # Q = FunctionSpace(mesh, "CG", p+1); bcmethod = "geometric"
    # Q = FunctionSpace(mesh, "DG", p); bcmethod = "geometric"

Z = V * Q
z = Function(Z, name="sln")
u, lam = split(z)
test = TestFunction(Z)

n = FacetNormal(mesh)
def cr(u):
    du = grad(u)
    return as_vector([du[0, 0] - du[1, 1], du[0,1] + du[1, 0]])
def intdx(u):
    return inner(u, u) * dx

E = 1e-10 * intdx(u) \
    + 1e-2 * intdx(grad(u)) \
    # + 1e0 * intdx(cr(u))
if zeroTangent:
    # E += inner(u-n, lam) * ds
    E += (inner(u,n)-g)*lam[0] * ds + inner(u,t)*lam[1] * ds
else:
    E += (inner(u, n)-g) * lam * ds

bc = DirichletBC(Z.sub(1), 1, "on_boundary")
exteriorNodes = Z.sub(1).boundary_nodes("on_boundary", bcmethod)
interiorNodes = [i for i in range(Z.sub(1).node_count) if i not in exteriorNodes]
bc.nodes = interiorNodes
print("Z.sub(0) exterior nodes", len(Z.sub(0).boundary_nodes("on_boundary", "geometric")))
print("Z.sub(1) exterior nodes", len(exteriorNodes))

F = derivative(E, z)

comm = mesh.comm
if comm.size == 1:
    import numpy as np
    J = derivative(derivative(E, z), z)
    A = assemble(J)
    A.force_evaluation()
    M = A.M[0, 1].handle[:, :]
    sigma = np.linalg.svd(M, compute_uv=False)
    rank = np.where(sigma>1e-8)[0].size
    print("Rank of constraint matrix:", rank)
    if rank < len(exteriorNodes):
        print("Rank of contraint matrix to small, add stabilization")
        E += 1e-10 * inner(lam, lam) * ds

else:
    E += 1e-10 * inner(lam, lam) * ds

F = derivative(E, z)

sp = {
    "mat_type": "aij",
    "snes_monitor": None,
    "ksp_type": "preonly",
    "pc_type": "lu",
    "pc_factor_mat_solver_type": "mumps",
    "ksp_converged_reason": None,
    "mat_mumps_icntl_14": 300,
}
solve(F == 0, z, bcs=bc, solver_parameters=sp)

out = File("out.pvd")
outfn = Function(Z.sub(0), name="sln").interpolate(z.split()[0])
out.write(outfn)

print("||u-g*n|| =", assemble(inner(u-g*n, u-g*n) * ds)**0.5)
print("||u·n-g|| =", assemble((inner(u,n)-g)**2 * ds)**0.5)
# print("||u·t||   =", assemble(inner(u,t)**2 * ds)**0.5)


Tvec = mesh.coordinates.vector()
Tvec += Function(VectorFunctionSpace(mesh, "CG", 1)).interpolate(z.split()[0]).vector()
out.write(outfn.assign(0))
