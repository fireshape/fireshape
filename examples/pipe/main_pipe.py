import firedrake as fd
import fireshape as fs
import fireshape.zoo as fsz
import ROL
from PDEconstraint_pipe import NavierStokesSolver
from objective_pipe import PipeObjective

# setup problem
mesh = fd.Mesh("pipe.msh")
Q = fs.FeControlSpace(mesh)
inner = fs.LaplaceInnerProduct(Q, fixed_bids=[10, 11, 12])
q = fs.ControlVector(Q, inner)

# setup PDE constraint
if mesh.topological_dimension() == 2:   #in 2D
    viscosity = fd.Constant(1./400.)
elif mesh.topological_dimension() == 3: #in 3D
    raise AssertionError('3D geometry has holes, fix it')
    viscosity = fd.Constant(1/10.) #simpler problem in 3D
else:
    raise NotImplementedError
e = NavierStokesSolver(Q.mesh_m, viscosity)

# save state variable evolution in file u.pvd
#e.solve()
out = fd.File("solution/u.pvd")
def cb(): return out.write(e.solution.split()[0])

# create PDEconstrained objective functional
J_ = PipeObjective(e, Q, cb=cb)
J = fs.ReducedObjective(J_, e)

# add regularization to improve mesh quality
Jq = fsz.MoYoSpectralConstraint(10, fd.Constant(0.5), Q) #this fails in a funny way
J = J + Jq

# Set up volume constraint
vol = fsz.VolumeFunctional(Q)
initial_vol = vol.value(q, None)
econ = fs.EqualityConstraint([vol], target_value=[initial_vol])
emul = ROL.StdVector(1)

# ROL parameters
params_dict = {
'General': {'Print Verbosity':1, #set to 1 if you struggle to understand the output
            'Secant': {'Type': 'Limited-Memory BFGS', 'Maximum Storage': 10}},
'Step': {'Type': 'Augmented Lagrangian',
         'Augmented Lagrangian': {'Subproblem Step Type': 'Trust Region',
                                   'Print Intermediate Optimization History': True,
                                   #'Subproblem Iteration Limit': 5}},
                                   'Subproblem Iteration Limit': 5}},
                                   #'Subproblem Iteration Limit': 10}}, #this fails with nans in computing grad
                                                                       #observation: a lot of subits lead to compressing
                                                                       #nodes in the middle of the pipe
'Status Test': {'Gradient Tolerance': 1e-2,
                'Step Tolerance': 1e-2,
                'Constraint Tolerance': 1e-1,
                'Iteration Limit': 2} #we can raise this to 100, nothing changes and it doesn't crash, it's good news, but finding appropriate stopping criteria is challenging
                }
params = ROL.ParameterList(params_dict, "Parameters")
problem = ROL.OptimizationProblem(J, q, econ=econ, emul=emul)
solver = ROL.OptimizationSolver(problem, params)
solver.solve()
