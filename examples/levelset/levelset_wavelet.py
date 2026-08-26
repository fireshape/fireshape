import firedrake as fd
import fireshape.zoo as fsz
import fireshape as fs
import ROL

mesh = fs.DiskMesh(0.1)

bbox = [(-1.01, 1.01), (-1.01, 1.01)]
primal_orders = [3, 3]
dual_orders = [3, 3]
levels = [4, 4]
Q = fs.WaveletControlSpace(mesh, bbox, primal_orders, dual_orders, levels,
                           homogeneous_bc=[False, False])
inner = fs.H1InnerProduct(Q)
Q.assign_inner_product(inner)
q = fs.ControlVector(Q, inner)

mesh_m = Q.mesh_m
(x, y) = fd.SpatialCoordinate(mesh_m)

f = (pow(x, 2))+pow(2*y, 2) - 1
out = fd.File("domain.pvd")

J = fsz.LevelsetFunctional(f, Q, cb=lambda: out.write(mesh_m.coordinates))
J = 0.1 * J

g = q.clone()
J.gradient(g, q, None)
J.checkGradient(q, g, 4, 1)

g.scale(1 / g.norm())
Q.visualize_control(g)

params_dict = {
    'Step': {
        'Type': 'Line Search',
        'Line Search': {
            'Descent Method': {
                'Type': 'Quasi-Newton Step'
            }
        }
    },
    'General': {
        'Secant': {
            'Type': 'Limited-Memory BFGS',
            'Maximum Storage': 25
        }
    },
    'Status Test': {
        'Gradient Tolerance': 1e-4,
        'Step Tolerance': 1e-10,
        'Iteration Limit': 30
    }
}

params = ROL.ParameterList(params_dict, "Parameters")
problem = ROL.OptimizationProblem(J, q)
solver = ROL.OptimizationSolver(problem, params)
solver.solve()
