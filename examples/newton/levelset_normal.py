from firedrake import SpatialCoordinate, sqrt
import firedrake as fd
import fireshape as fs
import fireshape.zoo as fsz
import ROL

mesh = fs.DiskMesh(0.1)
Q = fs.FeScalarControlSpace(mesh, allow_tangential=True)
inner = fs.SurfaceInnerProduct(Q)

mesh_m = Q.mesh_m

(x, y) = SpatialCoordinate(mesh_m)
a = 0.8
b = 2.0
f = (sqrt((x - a)**2 + b * y**2) - 1) \
* (sqrt((x + a)**2 + b * y**2) - 1) \
* (sqrt(b * x**2 + (y - a)**2) - 1) \
* (sqrt(b * x**2 + (y + a)**2) - 1) - 0.01



q = fs.ControlVector(Q, inner)
out = fd.File("domain.pvd")

J = fsz.LevelsetFunctional(f, Q)
g = q.clone()
J.update(q, None, 1)
J.gradient(g, q, None)
g.scale(-0.3)
J.update(g, None, 1)
J.checkGradient(q, g, 9, 1)
J.checkHessVec(q, g, 7, 1)

params_dict = {
    'General':{
        'Krylov':
        {
            'Type': 'Conjugate Gradient',
            'Iteration Limit': 100,
            'Use Initial Guess': True
        },
        'Secant': {
            'Type': 'Limited-Memory BFGS',
            'Maximum Storage': 10,
            'Use as Preconditioner': True
        },
    },
    'Step': {
        'Type': 'Line Search',
        'Line Search': {
            'Descent Method': {
                'Type': 'Newton-Krylov'
                # 'Type': 'Quasi-Newton Method'
            }
        }
    },
    'Status Test': {
        'Gradient Tolerance': 1e-11,
        'Step Tolerance': 1e-10,
        'Iteration Limit': 20
    }
}

params = ROL.ParameterList(params_dict, "Parameters")
problem = ROL.OptimizationProblem(J, q)
solver = ROL.OptimizationSolver(problem, params)
gradient_norms = []
def cb():
    J.gradient(g, None, None)
    gradient_norms.append(g.norm())
    out.write(mesh_m.coordinates)
J.cb = cb
solver.solve()
import matplotlib.pyplot as plt
plt.figure()
plt.semilogy([gradient_norms[i+1]/gradient_norms[i] for i in range(len(gradient_norms)-1)])
# plt.semilogy([gradient_norms[i+1]/gradient_norms[i]**2 for i in range(len(gradient_norms)-1)])
plt.show()
J.gradient(g, q, None)
J.checkGradient(q, g, 9, 1)
