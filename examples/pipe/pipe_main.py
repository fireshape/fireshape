import firedrake as fd
import fireshape as fs
import ROL
from pipe_PDEconstraint import NavierStokesSolver
from pipe_objective import PipeObjective

#setup problem
mesh = fd.Mesh("pipe.msh")

Q = fs.FeControlSpace(mesh)
inner = fs.LaplaceInnerProduct(Q, fixed_bids=[1, 2, 3])
q = fs.ControlVector(Q, inner)

#setup PDE constraint
e = NavierStokesSolver(Q.mesh_m)

#save state variable evolution in file u.pvd
e.solve()
out = fd.File("u.pvd")
cb = lambda: out.write(e.solution.split()[0])
cb()

#create PDEconstrained objective functional
J_ = PipeObjective(e, Q, cb=cb)
J = fs.ReducedObjective(J_, e)

#volume constraint
class VolumeFunctional(fs.ShapeObjective):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        #physical mesh
        self.mesh_m = self.Q.mesh_m

    def value_form(self):
        #volume integral
        return fd.Constant(1.0) * fd.dx(domain=self.mesh_m)
vol = VolumeFunctional(Q)
initial_vol = vol.value(q, None)
econ = fs.EqualityConstraint([vol], target_value=[initial_vol])
emul = ROL.StdVector(1)

#ROL parameters
params_dict = {
    'General': {
        'Secant': {'Type': 'Limited-Memory BFGS',
                   'Maximum Storage': 5}},
    'Step': {
        'Type': 'Augmented Lagrangian',
        'Line Search': {'Descent Method': {
            'Type': 'Quasi-Newton Step'}
        },
        'Augmented Lagrangian': {
            'Subproblem Step Type': 'Line Search',
            'Penalty Parameter Growth Factor': 1.05,
            'Print Intermediate Optimization History': True,
            'Subproblem Iteration Limit': 5
        }},
    'Status Test': {
        'Gradient Tolerance': 1e-4,
        'Step Tolerance': 1e-3,
        'Iteration Limit': 6}
}
params = ROL.ParameterList(params_dict, "Parameters")
problem = ROL.OptimizationProblem(J, q, econ=econ, emul=emul)
solver = ROL.OptimizationSolver(problem, params)
solver.solve()
print(vol.value(q, None) - initial_vol)
