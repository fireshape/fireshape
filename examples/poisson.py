import firedrake as fd
import fireshape as fs
import fireshape.zoo as fsz

import ROL

mesh = fd.UnitSquareMesh(100,100)
inner = fs.ElasticityInnerProduct()
Q = fs.FeControlSpace(mesh, inner)
mesh_m = Q.mesh_m
e = fsz.PoissonSolver(mesh_m)
e.solve()
out = fd.File("u.pvd")

def cb(*args):
    out.write(e.solution.split()[0])
cb()
e = fsz.L2trackingOjective(e, Q, cb=cb)#, scale=0.001)
Jr = fs.ReducedObjective(J, e)
q = fs.ControlVector(Q)
print(Jr.value(q, None))
g = q.clone()

#Jr.gradient(g, q, None)
#Jr.checkGradient(q, g, 5, 1)

params_dict = {
        'General': {
            'Secant': { 'Type': 'Limited-Memory BFGS', 'Maximum Storage': 25 } },
            'Step': {
                'Type': 'Augmented Lagrangian',
                'Line Search': {'Descent Method': {
                    'Type': 'Quasi-Newton Step'}
                    },
                'Augmented Lagrangian': {
                    'Subproblem Step Type': 'Line Search',
                    'Penalty Parameter Growth Factor': 2.,
                    'Print Intermediate Optimization History': True
                    }},
        'Status Test': {
            'Gradient Tolerance': 1e-15, 'Relative Gradient Tolerance': 1e-10,
            'Step Tolerance': 1e-16, 'Relative Step Tolerance': 1e-10,
            'Iteration Limit': 3}
        }

params = ROL.ParameterList(params_dict, "Parameters")
problem = ROL.OptimizationProblem(Jr, q)
solver = ROL.OptimizationSolver(problem, params)
solver.solve()
