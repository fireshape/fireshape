import firedrake as fd
import fireshape as fs
import _ROL as ROL
import math

n = 5
mesh = fd.UnitSquareMesh(n, n)
mesh = fd.Mesh(fd.Function(fd.VectorFunctionSpace(mesh, "CG", 2)).interpolate(fd.SpatialCoordinate(mesh)))


inner = fs.LaplaceInnerProduct(mesh)
Q = fs.FeMultiGridControlSpace(mesh, inner)
mesh_m = Q.moving_mesh()
V_m = fd.FunctionSpace(mesh_m, "CG", 1)
f_m = fd.Function(V_m)

out = fd.File("mesh_m.pvd")
out.write(f_m)

q = fs.ControlVector(Q)
q.fun.interpolate(fd.Expression(("0", "x[0]*x[0]")))
Q.update_mesh(q)
out.write(f_m)
q = fs.ControlVector(Q)
(x, y) = fd.SpatialCoordinate(mesh_m)
f = (pow(x-0.5, 2))+pow(y-0.5, 2) - 2.

class LevelsetFunctional(fs.Objective):

    def val(self):
        return fd.assemble(f * fd.dx)

    def derivative_form(self, v):
        return fd.div(f*v) * fd.dx

# out = fd.File("domain.pvd")
out = fd.File("T.pvd")
J = LevelsetFunctional(q, cb=lambda: out.write(Q.T_fine))
# J = LevelsetFunctional(q, cb=lambda: out.write(q.domain().coordinates))

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
x = q.clone() # we need a seperate iteration vector
problem = ROL.OptimizationProblem(J, x)
solver = ROL.OptimizationSolver(problem, params)
solver.solve()
