import firedrake as fd
import fireshape as fs
import fireshape.zoo as fsz

import ROL

mesh = fd.Mesh("Sphere2D.msh")

# Q = fs.FeControlSpace(mesh)
mh = fd.MeshHierarchy(mesh, 1)
Q = fs.FeMultiGridControlSpace(mh, coarse_control=True)
inner = fs.ElasticityInnerProduct(Q, fixed_bids=[1, 2, 3])
mesh_m = Q.mesh_m
(x, y) = fd.SpatialCoordinate(mesh_m)
inflow_expr = fd.Constant((1.0, 0.0))
e = fsz.StokesSolver(mesh_m, inflow_bids=[1, 2],
                     inflow_expr=inflow_expr, noslip_bids=[4])
e.solve()
out = fd.VTKFile("u.pvd")


def cb(*args):
    u, p = e.solution.subfunctions
    out.write(u)


cb()

Je = fsz.EnergyObjective(e, Q, cb=cb)
Jr = 1e-2 * fs.ReducedObjective(Je, e)
Js = fsz.MoYoSpectralConstraint(10., fd.Constant(0.7), Q)
Jd = 1e-3 * fsz.DeformationRegularization(Q, l2_reg=1e-2, sym_grad_reg=1e0,
                                          skew_grad_reg=1e-2)
J = Jr + Jd + Js
q = fs.ControlVector(Q, inner)
g = q.clone()

vol = fsz.LevelsetFunctional(fd.Constant(1.0), Q)
baryx = fsz.LevelsetFunctional(x, Q)
baryy = fsz.LevelsetFunctional(y, Q)
econ = fs.EqualityConstraint([vol, baryx, baryy])
emul = ROL.StdVector(3)

params_dict = {
    'General': {
        'Secant': {
            'Type': 'Limited-Memory BFGS', 'Maximum Storage': 5
        }
    },
    'Step': {
        'Type': 'Augmented Lagrangian',
        'Line Search': {
            'Descent Method': {
                'Type': 'Quasi-Newton Step'
            }
        },
        'Augmented Lagrangian': {
            'Subproblem Step Type': 'Line Search',
            'Penalty Parameter Growth Factor': 2.,
            'Print Intermediate Optimization History': True,
            'Subproblem Iteration Limit': 20
        }
    },
    'Status Test': {
        'Gradient Tolerance': 1e-4,
        'Step Tolerance': 1e-5,
        'Iteration Limit': 4
    }
}
params = ROL.ParameterList(params_dict, "Parameters")
problem = ROL.OptimizationProblem(J, q, econ=econ, emul=emul)
solver = ROL.OptimizationSolver(problem, params)
vol_before = vol.value(q, None)
solver.solve()
print(vol.value(q, None)-vol_before)
