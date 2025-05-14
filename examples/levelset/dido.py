from firedrake import *
from fireshape import *
import fireshape.zoo as fsz
import ROL


class NegativeArea(PDEconstrainedObjective):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def objective_value(self):
        return assemble(Constant(-1) * dx(self.Q.mesh_m))


# Select initial guess, control space, and inner product
mesh = UnitSquareMesh(5, 5)
Q = FeControlSpace(mesh, add_to_degree_r=1)
IP = H1InnerProduct(Q)
q = ControlVector(Q, IP)

# Instantiate objective function J
out = VTKFile("domain.pvd")
J = NegativeArea(Q, cb=lambda: out.write(Q.mesh_m.coordinates))

# Set up perimeter constraint
perimeter = fsz.SurfaceAreaFunctional(Q)
initial_perimeter = perimeter.value(q, None)
econ = EqualityConstraint([perimeter], target_value=[initial_perimeter])
emul = ROL.StdVector(1)

# Select the optimization algorithm and solve the problem
pd = {'General': {'Print Verbosity': 0,
                  'Secant': {'Type': 'Limited-Memory BFGS',
                             'Maximum Storage': 10}},
      'Step': {'Type': 'Augmented Lagrangian',
               'Augmented Lagrangian':
               {'Use Default Problem Scaling': False,
                'Constraint Scaling': 1.5,
                'Subproblem Step Type': 'Trust Region',
                'Print Intermediate Optimization History': False,
                'Subproblem Iteration Limit': 10}},
      'Status Test': {'Gradient Tolerance': 1e-2,
                      'Step Tolerance': 1e-3,
                      'Constraint Tolerance': 1e-1,
                      'Iteration Limit': 10}}
params = ROL.ParameterList(pd, "Parameters")
problem = ROL.OptimizationProblem(J, q, econ=econ, emul=emul)
solver = ROL.OptimizationSolver(problem, params)
solver.solve()
