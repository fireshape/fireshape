import unittest
import firedrake as fd
import fireshape as fs
import fireshape.zoo as fsz

import ROL


class EqualityConstraintTest(unittest.TestCase):

    def test_equality_constraint(self):
        n = 100
        mesh = fd.UnitSquareMesh(n, n)

        inner = fs.LaplaceInnerProduct()
        Q = fs.FeControlSpace(mesh, inner)
        mesh_m = Q.mesh_m
        (x, y) = fd.SpatialCoordinate(mesh_m)

        q = fs.ControlVector(Q)
        #out = fd.File("T.pvd") # commented to stop storing vtu files

        #def cb(*args):
        #    out.write(Q.T)

        #cb()
        f = (pow(x-0.5, 2))+pow(y-0.5, 2) - 1.2

        J = fsz.LevelsetFunctional(f, Q)#, cb=cb)
        vol = fsz.LevelsetFunctional(fd.Constant(1.0), Q)
        e = fs.EqualityConstraint([vol])
        emul = ROL.StdVector(1)

        params_dict = {
            'General': {
                'Secant': {'Type': 'Limited-Memory BFGS',
                           'Maximum Storage': 25}},
            'Step': {
                'Type': 'Augmented Lagrangian',
                'Line Search': {'Descent Method': {
                    'Type': 'Quasi-Newton Step'}},
                'Augmented Lagrangian': {
                    'Subproblem Step Type': 'Line Search',
                    'Penalty Parameter Growth Factor': 5.,
                    'Initial Penalty Parameter' : 1.
                    }},
            'Status Test': {
                'Gradient Tolerance': 1e-7,
                'Relative Gradient Tolerance': 1e-6,
                'Step Tolerance': 1e-10, 'Relative Step Tolerance': 1e-10,
                'Iteration Limit': 150}}

        params = ROL.ParameterList(params_dict, "Parameters")
        problem = ROL.OptimizationProblem(J, q, econ=e, emul=emul)
        solver = ROL.OptimizationSolver(problem, params)
        solver.solve()



if __name__ == '__main__':
    unittest.main()
