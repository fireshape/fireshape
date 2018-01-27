import firedrake as fd
import fireshape as fs
import fireshape.zoo as fsz

import _ROL as ROL

mesh = fd.Mesh("Sphere2D.msh")

inner = fs.ElasticityInnerProduct(fixed_bids=[1, 2, 3])
# Q = fs.FeControlSpace(mesh, inner)
Q = fs.FeMultiGridControlSpace(mesh, inner, refinements=2, order=2)
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

J = fsz.EnergyObjective(e, Q, cb=cb, scale=0.001)
Jr = fs.ReducedObjective(J, e)
q = fs.ControlVector(Q)
print(Jr.value(q, None))
g = q.clone()

Jr.gradient(g, q, None)
Jr.checkGradient(q, g, 5, 1)

vol = fsz.LevelsetFunctional(fd.Constant(1.0), Q, scale=2.)
econ = fs.EqualityConstraint([vol])
emul = ROL.StdVector(1)

params_dict = {
        'General': {
            'Secant': { 'Type': 'Limited-Memory BFGS', 'Maximum Storage': 25 } },
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
problem = ROL.OptimizationProblem(Jr, q, econ=[econ], emul=[emul])
solver = ROL.OptimizationSolver(problem, params)
vol_before = vol.value(q, None)
solver.solve()
print(vol.value(q, None)-vol_before)
