import firedrake as fd
import fireshape as fs
import fireshape.zoo as fsz

import _ROL as ROL

mesh = fd.Mesh("Sphere2D.msh")

inner = fs.LaplaceInnerProduct(fixed_bids=[1, 2, 3])
# Q = fs.FeControlSpace(mesh, inner)
Q = fs.FeMultiGridControlSpace(mesh, inner, refinements_per_level=2)
mesh_m = Q.mesh_m
(x, y) = fd.SpatialCoordinate(mesh_m)
inflow_expr = fd.Constant((1.0, 0.0)) * (2-y) * (2+y)
e = fsz.StokesSolver(mesh_m, inflow_bids=[2],
                     inflow_expr=inflow_expr, noslip_bids=[1, 4])
e.solve()
out = fd.File("u.pvd")


def cb(*args):
    out.write(e.solution.split()[0])


cb()

J = fsz.EnergyObjective(e, Q, cb=cb, scale=2e-4)
Jr = fs.ReducedObjective(J, e)
q = fs.ControlVector(Q)
print(Jr.value(q, None))

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
            'Iteration Limit': 1}
        }

params = ROL.ParameterList(params_dict, "Parameters")
problem = ROL.OptimizationProblem(Jr, q)
solver = ROL.OptimizationSolver(problem, params)
solver.solve()
