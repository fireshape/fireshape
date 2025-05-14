from firedrake import *
from fireshape import *
import ROL


class LevelsetFunction(PDEconstrainedObjective):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # physical mesh
        mesh_m = self.Q.mesh_m

        # integrand defined in terms of physical coordinates
        x, y, z = SpatialCoordinate(mesh_m)
        self.f = (x - 0.5)**2 + (y - 0.5)**2 + (z - 0.5)**2 - 0.64

    def objective_value(self):
        return assemble(self.f * dx)


# Select initial guess, control space, and inner product
mesh = UnitBallMesh(refinement_level=3)
Q = FeControlSpace(mesh)
IP = H1InnerProduct(Q)
q = ControlVector(Q, IP)

# Instantiate objective function J
out = VTKFile("domain.pvd")
J = LevelsetFunction(Q, cb=lambda: out.write(Q.mesh_m.coordinates))

# Select the optimization algorithm and solve the problem
pd = {'Step': {'Type': 'Trust Region'},
      'General': {'Secant': {'Type': 'Limited-Memory BFGS',
                                     'Maximum Storage': 25}},
      'Status Test': {'Gradient Tolerance': 1e-3,
                      'Step Tolerance': 1e-8,
                      'Iteration Limit': 30}}
params = ROL.ParameterList(pd, "Parameters")
problem = ROL.OptimizationProblem(J, q)
solver = ROL.OptimizationSolver(problem, params)
solver.solve()
