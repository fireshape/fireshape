from firedrake import *
from fireshape import *
import numpy as np
import ROL

d = 3

# goal: update the mesh coordinates of a periodic mesh
# so that a function defined on the mesh becomes a given one
if d == 2:
    mesh = PeriodicUnitSquareMesh(30, 30)
elif d== 3:
    mesh = PeriodicUnitCubeMesh(20, 20, 20)
Q = PeriodicControlSpace(mesh)  #how can we fix the boundary?
#inner = LaplaceInnerProduct(Q)
inner = ElasticityInnerProduct(Q)
#inner = H1InnerProduct(Q)
q = ControlVector(Q, inner)

# save shape evolution in file domain.pvd
V = FunctionSpace(Q.mesh_m, "DG", 0)
sigma = Function(V)
if d == 2:
    x, y = SpatialCoordinate(Q.mesh_m)
    g = sin(y*np.pi)  # truncate at bdry
    f = cos(2*np.pi*x)*g
    perturbation = 0.05*sin(x*np.pi)*g**2
    sigma.interpolate(g*cos(2*np.pi*x*(1+perturbation)))
elif d == 3:
    x, y, z = SpatialCoordinate(Q.mesh_m)
    g = sin(y*np.pi)*sin(z*np.pi)  # truncate at bdry
    f = cos(2*np.pi*x)*g
    perturbation = 0.05*sin(x*np.pi)*g**2
    sigma.interpolate(g*cos(2*np.pi*x*(1+perturbation)))

class LevelsetFct(ShapeObjective):
    def __init__(self, sigma, f, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.sigma = sigma  #initial
        self.f = f          #target
        Vdet = FunctionSpace(Q.mesh_r, "DG", 0)
        self.detDT = Function(Vdet)

    def value_form(self):
        # volume integral
        self.detDT.interpolate(det(grad(self.Q.T)))
        if min(self.detDT.vector()) > 0.05:
            integrand = (self.sigma - self.f)**2
        else:
            integrand = np.nan*(self.sigma - self.f)**2
        return integrand * dx(metadata={"quadrature_degree":1})

CB = File("domain.pvd")
J = LevelsetFct(sigma, f, Q, cb=lambda: CB.write(sigma))

# ROL parameters
params_dict = {'Step': {'Type': 'Trust Region'},
'General': {'Secant': {'Type': 'Limited-Memory BFGS',
                       'Maximum Storage': 25}},
'Status Test': {'Gradient Tolerance': 1e-4,
                'Step Tolerance': 1e-4,
                'Iteration Limit': 60}}

params = ROL.ParameterList(params_dict, "Parameters")
problem = ROL.OptimizationProblem(J, q)
solver = ROL.OptimizationSolver(problem, params)
solver.solve()
