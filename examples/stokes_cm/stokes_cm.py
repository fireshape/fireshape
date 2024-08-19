import firedrake as fd
import fireshape as fs
import fireshape.zoo as fsz

import ROL

mesh_c = fd.Mesh("stoke_control.msh")
mesh_r = fd.Mesh("stoke_hole.msh")

# Q = fs.FeControlSpace(mesh)
# Q = fs.FeMultiGridControlSpace(mesh, refinements=1, degree=1)

S = fd.FunctionSpace(mesh_c, "DG", 0)
I = fd.Function(S, name="indicator")
x = fd.SpatialCoordinate(mesh_c)

I.interpolate(fd.conditional(x[0]**2 + x[1]**2 < 0.5**2, 1, 0))

Q = fs.HelmholtzControlSpace(mesh_c, mesh_r, I, 25, 4)
# this 1,2,3 represents the physical curve tag 
# have added 5 here as this is the complete outside bit
inner = fs.L2InnerProduct(Q, fixed_bids=[1, 2, 3])
mesh_m = Q.mesh_m

(x, y) = fd.SpatialCoordinate(mesh_m)
inflow_expr = fd.Constant((1.0, 0.0))
e = fsz.StokesSolver(mesh_m, inflow_bids=[1, 2],
                     inflow_expr=inflow_expr, noslip_bids=[4], direct=False)
e.solve()

out = fd.File("no_regularization/moved.pvd")
out2 = fd.File("no_regularization/control.pvd")

control_copy = Q.mesh_c.coordinates.copy(deepcopy=True)
def cb(*args):
    out.write(e.solution.split()[0])
    Q.mesh_c.coordinates.assign(Q.mesh_c.coordinates + Q.dphi)
    out2.write(Q.mesh_c.coordinates, I, Q.dphi)
    Q.mesh_c.coordinates.assign(control_copy)


cb()
Je = fsz.EnergyObjective(e, Q, cb=cb)
Jr = 1e-2 * fs.ReducedObjective(Je, e)
# Js = fsz.MoYoSpectralConstraint(10., fd.Constant(0.7), Q)
# Jd = 1e-3 * fsz.DeformationRegularization(Q, l2_reg=1e-2, sym_grad_reg=1e0, skew_grad_reg=1e-2)
# Tryu removing Js and check values of funcionals in ours vs theirs
J = Jr
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
            'Subproblem Iteration Limit': 50
        }
    },
    'Status Test': {
        'Gradient Tolerance': 1e-4,
        'Step Tolerance': 1e-5,
        'Iteration Limit': 20
    }
}
params = ROL.ParameterList(params_dict, "Parameters")
problem = ROL.OptimizationProblem(J, q, econ=econ, emul=emul)
solver = ROL.OptimizationSolver(problem, params)
vol_before = vol.value(q, None)
solver.solve()
print(vol.value(q, None)-vol_before)