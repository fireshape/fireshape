from firedrake import *
from fireshape import *
import numpy as np
import ROL

# setup problem
#mesh = PeriodicSquareMesh(10, 10, 1, diagonal="crossed")
mesh = UnitSquareMesh(30, 30)
Q = FeControlSpace(mesh)
inner = LaplaceInnerProduct(Q, fixed_bids=[1,2,3,4])#"on_boundary")
q = ControlVector(Q, inner)

# save shape evolution in file domain.pvd
V = FunctionSpace(Q.mesh_m, "DG", 1)
sigma = Function(V)
x, y = SpatialCoordinate(Q.mesh_m)
perturbation = 0.1*sin(x*np.pi)*(16*y**2*(1-y)**2)
sigma.interpolate(y*(1-y)*(cos(2*np.pi*x*(1+perturbation))))
sigma.interpolate(y*(1-y)*(cos(2*np.pi*x*(1+0.1*16*y**2*(1-y)**2*sin(x*np.pi)))))
f = cos(2*np.pi*x)*y*(1-y)

class LevelsetFct(ShapeObjective):
    def __init__(self, sigma, f, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.sigma = sigma
        self.f = f
        Vdet = FunctionSpace(Q.mesh_r, "DG", 0)
        self.detDT = Function(Vdet)

    def value_form(self):
        # volume integral
        self.detDT.interpolate(det(grad(self.Q.T)))
        if min(self.detDT.vector()) > 0.05:
            return (self.sigma - self.f)**2 * dx(metadata={"quadrature_degree":1})
        else:
            return np.nan*(self.sigma - self.f)**2 * dx(metadata={"quadrature_degree":1})

CB = File("domain.pvd")
J = LevelsetFct(sigma, f, Q, cb=lambda: CB.write(sigma))


# ROL parameters
params_dict = {'Step': {'Type': 'Trust Region'},
'General': {'Secant': {'Type': 'Limited-Memory BFGS',
                       'Maximum Storage': 25}},
'Status Test': {'Gradient Tolerance': 1e-4,
                'Step Tolerance': 1e-4,
                'Iteration Limit': 30}}

params = ROL.ParameterList(params_dict, "Parameters")
problem = ROL.OptimizationProblem(J, q)
solver = ROL.OptimizationSolver(problem, params)
solver.solve()
