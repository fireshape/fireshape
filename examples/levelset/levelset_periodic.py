from firedrake import fd
from fireshape import fs
import numpy as np
import ROL

d = 3

# goal: update the mesh coordinates of a periodic mesh
# so that a function defined on the mesh becomes a given one
if d == 2:
    mesh = fd.PeriodicUnitSquareMesh(30, 30)
elif d == 3:
    mesh = fd.PeriodicUnitCubeMesh(20, 20, 20)
Q = fs.PeriodicControlSpace(mesh)
# inner = LaplaceInnerProduct(Q)
inner = fs.ElasticityInnerProduct(Q)
# inner = H1InnerProduct(Q)
q = fs.ControlVector(Q, inner)

# save shape evolution in file domain.pvd
V = fd.FunctionSpace(Q.mesh_m, "DG", 0)
sigma = fd.Function(V)
if d == 2:
    x, y = fd.SpatialCoordinate(Q.mesh_m)
    g = fd.sin(y*np.pi)  # truncate at bdry
    f = fd.cos(2*np.pi*x)*g
    perturbation = 0.05*fd.sin(x*np.pi)*g**2
    sigma.interpolate(g*fd.cos(2*np.pi*x*(1+perturbation)))
elif d == 3:
    x, y, z = fd.SpatialCoordinate(Q.mesh_m)
    g = fd.sin(y*np.pi)*fd.sin(z*np.pi)  # truncate at bdry
    f = fd.cos(2*np.pi*x)*g
    perturbation = 0.05*fd.sin(x*np.pi)*g**2
    sigma.interpolate(g*fd.cos(2*np.pi*x*(1+perturbation)))


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
        else:
            integrand = np.nan*(self.sigma - self.f)**2
        return integrand * fd.dx(metadata={"quadrature_degree": 1})


CB = fd.File("domain.pvd")
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
