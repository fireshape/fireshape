import firedrake as fd
import fireshape as fs
import fireshape.zoo as fsz
import ROL
from PDEconstraint_pipe import NavierStokesSolver
from objective_pipe import PipeObjective
from icecream import install
install()

# setup problem
mesh_r = fd.Mesh("pipe.msh", name="mesh_r")
mesh_c = fd.Mesh("pipe_control.msh", name="mesh_c")

S = fd.FunctionSpace(mesh_c, "DG", 0)
I = fd.Function(S, name="indicator")
fd.par_loop(("{[i] : 0 <= i < f.dofs}", "f[i, 0] = 1.0"),
         fd.dx(2),
         {"f": (I, fd.WRITE)})

Q = fs.HelmholtzControlSpace(mesh_c, mesh_r, I, 25, (10, 11, 12, 13), interior_bc_tags=[10, 11, 12])

inner = fs.LaplaceInnerProduct(Q, fixed_bids=[10, 11, 12])
q = fs.ControlVector(Q, inner)

# setup PDE constraint
viscosity = fd.Constant(1./400.)
e = NavierStokesSolver(Q.mesh_m, viscosity)

# save state variable evolution in file u2.pvd or u3.pvd
out = fd.File("solution/no_deformation.pvd")

def cb():
    return out.write(e.solution.split()[0])

# create PDEconstrained objective functional
J_ = PipeObjective(e, Q, cb=cb)
J = fs.ReducedObjective(J_, e)

# add regularization to improve mesh quality
# Jq = fsz.MoYoSpectralConstraint(10, fd.Constant(0.5), Q)
# J = J

# Set up volume constraint
vol = fsz.VolumeFunctional(Q)
initial_vol = vol.value(q, None)
econ = fs.EqualityConstraint([vol], target_value=[initial_vol])
emul = ROL.StdVector(1)

# ROL parameters
params_dict = {
    'General': {'Print Verbosity': 0,  # set to 1 to understand output
                'Secant': {'Type': 'Limited-Memory BFGS',
                           'Maximum Storage': 10}},
    'Step': {'Type': 'Augmented Lagrangian',
             'Augmented Lagrangian':
             {'Subproblem Step Type': 'Trust Region',
              'Print Intermediate Optimization History': True,
              'Subproblem Iteration Limit': 15}},
    'Status Test': {'Gradient Tolerance': 1e-2,
                    'Step Tolerance': 1e-2,
                    'Constraint Tolerance': 1e-1,
                    'Iteration Limit': 15}}
params = ROL.ParameterList(params_dict, "Parameters")
problem = ROL.OptimizationProblem(J, q, econ=econ, emul=emul)
solver = ROL.OptimizationSolver(problem, params)
solver.solve()
