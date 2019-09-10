import firedrake as fd
import fireshape as fs
import firedrake_adjoint as fda
import fireshape.zoo as fsz
import ROL
from pipe_PDEconstraint import NavierStokesSolver
from pipe_objective import PipeObjective

# setup problem
mesh = fd.Mesh("pipe.msh")
Q = fs.FeControlSpace(mesh)
inner = fs.LaplaceInnerProduct(Q, fixed_bids=[10, 11, 12])
q = fs.ControlVector(Q, inner)

# setup PDE constraint
dim = mesh.topological_dimension()
if dim == 2:
    viscosity = fda.Constant(1./400.)
elif dim ==3:
    viscosity = fda.Constant(1/10.)
else:
    raise NotImplementedError
e = NavierStokesSolver(Q.mesh_m, viscosity)

# save state variable evolution in file u.pvd
e.solve()
out = fd.File("solution/u.pvd")
def cb(): return out.write(e.solution.split()[0])

# create PDEconstrained objective functional
J_ = PipeObjective(e, Q, cb=cb)
J = fs.ReducedObjective(J_, e)

# add regularization to improve mesh quality
Jq = fsz.MoYoSpectralConstraint(10, fd.Constant(0.5), Q)
J = J + Jq

# Set up volume constraint
vol = fsz.VolumeFunctional(Q)
initial_vol = vol.value(q, None)
econ = fs.EqualityConstraint([vol], target_value=[initial_vol])
emul = ROL.StdVector(1)

# ROL parameters
params_dict = {
'General': {'Secant': {'Type': 'Limited-Memory BFGS', 'Maximum Storage': 10}},
'Step': {'Type': 'Augmented Lagrangian',
         'Trust Region':{'Maximal Radius': 10},
         'Augmented Lagrangian': {'Subproblem Step Type': 'Trust Region',
                                   'Print Intermediate Optimization History': False,
                                   'Subproblem Iteration Limit': 20}},
'Status Test': {'Gradient Tolerance': 1e-2,
                'Step Tolerance': 1e-2,
                'Constraint Tolerance': 1e-1,
                'Iteration Limit': 5}
}
params = ROL.ParameterList(params_dict, "Parameters")
problem = ROL.OptimizationProblem(J, q, econ=econ, emul=emul)
solver = ROL.OptimizationSolver(problem, params)
solver.solve()
