import firedrake as fd
import fireshape as fs
import fireshape.zoo as fsz

import ROL
from math import pi

mesh = fd.Mesh("CircleInSquare.msh")

inner = fs.ElasticityInnerProduct(fixed_bids=[1])
Q = fs.FeControlSpace(mesh, inner)
# Q = fs.FeMultiGridControlSpace(mesh, inner, refinements=5, order=1)
mesh_m = Q.mesh_m
(x, y) = fd.SpatialCoordinate(mesh_m)
f= fd.sin(2*pi*x) * fd.sin(2*pi*y)
e = fsz.PoissonSolver(mesh_m, dirichlet_bids=[1, 2], dirichlet_vals=[0, 1], rhs=f)
e.solve()
out = fd.File("u.pvd")
outv = fd.File("v.pvd")


def cb(*args):
    out.write(e.solution)
    outv.write(e.solution_adj)

cb()

J = fsz.CoefficientIntegralFunctional(e.solution**2, Q, cb=cb, scale=1e0)
Jr = fs.ReducedObjective(J, e)
q = fs.ControlVector(Q)
print(Jr.value(q, None))
g = q.clone()

Jr.gradient(g, q, None)
Jr.checkGradient(q, g, 5, 1)
# import sys; sys.exit(0)

vol = fsz.LevelsetFunctional(fd.Constant(1.0), Q, scale=2.0)
econ = fs.EqualityConstraint([vol])
emul = ROL.StdVector(1)

params_dict = {
        'General': {
            'Secant': { 'Type': 'Limited-Memory BFGS', 'Maximum Storage': 2 } },
            'Step': {
                'Type': 'Augmented Lagrangian',
                'Line Search': {'Descent Method': {
                    'Type': 'Quasi-Newton Step'}
                    },
                'Augmented Lagrangian': {
                    'Subproblem Step Type': 'Line Search',
                    'Penalty Parameter Growth Factor': 2.,
                    'Print Intermediate Optimization History': True
                    }},
        'Status Test': {
            'Gradient Tolerance': 1e-15, 'Relative Gradient Tolerance': 1e-10,
            'Step Tolerance': 1e-16, 'Relative Step Tolerance': 1e-10,
            'Iteration Limit': 3}
        }

params = ROL.ParameterList(params_dict, "Parameters")
problem = ROL.OptimizationProblem(Jr, q, econ=econ, emul=emul)
solver = ROL.OptimizationSolver(problem, params)
vol_before = vol.value(q, None)
solver.solve()
print(vol.value(q, None)-vol_before)
