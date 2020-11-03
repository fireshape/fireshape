from firedrake import fd
from fireshape import fs
import numpy as np
import ROL

# goal: optimise the mesh coordinates so that
# a function defined on the mesh approximates
# a target one

# setup problem
mesh = fd.UnitSquareMesh(30, 30)
Q = fs.FeControlSpace(mesh)
inner = fs.LaplaceInnerProduct(Q, fixed_bids=[1, 2, 3, 4])
q = fs.ControlVector(Q, inner)

# save shape evolution in file domain.pvd
V = fd.FunctionSpace(Q.mesh_m, "DG", 1)
sigma = fd.Function(V)
x, y = fs.SpatialCoordinate(Q.mesh_m)
perturbation = 0.1*fd.sin(x*np.pi)*(16*y**2*(1-y)**2)
sigma.interpolate(y*(1-y)*(fd.cos(2*np.pi*x*(1+perturbation))))
f = fd.cos(2*np.pi*x)*y*(1-y)


class LevelsetFct(fs.ShapeObjective):
    def __init__(self, sigma, f, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.sigma = sigma  # initial
        self.f = f          # target
        Vdet = fd.FunctionSpace(Q.mesh_r, "DG", 0)
        self.detDT = fd.Function(Vdet)

    def value_form(self):
        # volume integral
        self.detDT.interpolate(fd.det(fd.grad(self.Q.T)))
        if min(self.detDT.vector()) > 0.05:
            integrand = (self.sigma - self.f)**2
            return integrand * fd.dx(metadata={"quadrature_degree": 1})
        else:
            integrand = np.nan * (self.sigma - self.f)**2
            return integrand * fd.dx(metadata={"quadrature_degree": 1})


CB = fd.File("domain.pvd")
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
