import unittest
import firedrake as fd
import fireshape as fs

import _ROL as ROL


class LevelsetTest(unittest.TestCase):

    def run_levelset_optimization(self, Q, write_output=False):
        mesh_m = Q.mesh_m
        (x, y) = fd.SpatialCoordinate(mesh_m)
        f = (pow(x-0.5, 2))+pow(y-0.5, 2) - 2.

        class LevelsetFunctional(fs.Objective):

            def value_form(self):
                return f * fd.dx

            def derivative_form(self, v):
                return fd.div(f*v) * fd.dx

        q = fs.ControlVector(Q)
        if write_output:
            out = fd.File("T.pvd")

            def cb(*args):
                out.write(Q.T)

            cb()
        else:
            cb = None
        J = LevelsetFunctional(Q, cb=cb)

        params_dict = {
            'General': {
                'Secant': {'Type': 'Limited-Memory BFGS',
                           'Maximum Storage': 25}},
            'Step': {
                'Type': 'Line Search',
                'Line Search': {'Descent Method': {
                    'Type': 'Quasi-Newton Step'}}},
            'Status Test': {
                'Gradient Tolerance': 1e-7,
                'Relative Gradient Tolerance': 1e-6,
                'Step Tolerance': 1e-10, 'Relative Step Tolerance': 1e-10,
                'Iteration Limit': 150}}

        params = ROL.ParameterList(params_dict, "Parameters")
        problem = ROL.OptimizationProblem(J, q)
        solver = ROL.OptimizationSolver(problem, params)
        solver.solve()
        state = solver.getAlgorithmState()
        self.assertTrue(state.gnorm < 1e-6)

    def test_fe(self):
        n = 100
        mesh = fd.UnitSquareMesh(n, n)

        inner = fs.LaplaceInnerProduct()
        Q = fs.FeControlSpace(mesh, inner)
        self.run_levelset_optimization(Q, write_output=False)

    def run_fe_mg(self, order, write_output=False):
        mesh = fd.UnitSquareMesh(4, 4)

        inner = fs.LaplaceInnerProduct()
        Q = fs.FeMultiGridControlSpace(mesh, inner, refinements=4, order=order)
        self.run_levelset_optimization(Q, write_output=write_output)

    def test_fe_mg_first_order(self):
        self.run_fe_mg(1, write_output=False)

    def test_fe_mg_second_order(self):
        self.run_fe_mg(2, write_output=False)


if __name__ == '__main__':
    unittest.main()
