import firedrake as fd
import fireshape as fs
import _ROL as ROL
import math

n = 30
# mesh = fd.UnitSquareMesh(n, n)
mesh = fd.Mesh("UnitSquareCrossed.msh")
mesh = fd.MeshHierarchy(mesh, 1, refinements_per_level=2)[-1]
fd.File("mesh_r.pvd").write(mesh.coordinates)
mesh = fd.Mesh(fd.Function(fd.VectorFunctionSpace(mesh, "CG", 1)).interpolate(fd.SpatialCoordinate(mesh)))


inner = fs.LaplaceInnerProduct(mesh)
Q = fs.FeMultiGridControlSpace(mesh, inner, refinements_per_level=4)
mesh_m = Q.mesh_m
V_m = fd.FunctionSpace(mesh_m, "CG", 1)
f_m = fd.Function(V_m)

(x, y) = fd.SpatialCoordinate(mesh_m)
f = (pow(x-0.5, 2))+pow(y-0.5, 2) - 2.

class LevelsetFunctional(fs.Objective):

    def val(self):
        return fd.assemble(f * fd.dx)

    def derivative_form(self, v):
        return fd.div(f*v) * fd.dx

q = fs.ControlVector(Q)
out = fd.File("T.pvd")
out.write(Q.T)
J = LevelsetFunctional(Q, cb=lambda: out.write(Q.T))

params_dict = {
    'General': {
        'Secant': { 'Type': 'Limited-Memory BFGS', 'Maximum Storage': 25 } },
    'Step': {
        'Type': 'Line Search',
        'Line Search': { 'Descent Method': { 'Type': 'Quasi-Newton Step' } }
    },
    'Status Test': {
        'Gradient Tolerance': 1e-15, 'Relative Gradient Tolerance': 1e-10,
        'Step Tolerance': 1e-16, 'Relative Step Tolerance': 1e-10,
        'Iteration Limit': 100 }
}

params = ROL.ParameterList(params_dict, "Parameters")
problem = ROL.OptimizationProblem(J, q)
solver = ROL.OptimizationSolver(problem, params)
solver.solve()
