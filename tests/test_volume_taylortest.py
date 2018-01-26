import unittest
import firedrake as fd
import fireshape as fs
import _ROL as ROL
import math

class VolumeTaylorTest(unittest.TestCase):

    def run_taylor_test(self, Q):
        mesh_m = Q.mesh_m
        f = fd.Constant(1.0)

        class LevelsetFunctional(fs.Objective):

            def val(self):
                return fd.assemble(f * fd.dx(domain=mesh_m))

            def derivative_form(self, v):
                return fd.div(f*v) * fd.dx(domain=mesh_m)

        x = fs.ControlVector(Q)
        J = LevelsetFunctional(Q)
        """ move mesh a bit to check that we are not doing the taylor test in T=id """
        g = x.clone()
        J.gradient(g, x, None)
        x.plus(g)
        J.update(x, None, 1)

        """ Start taylor test """
        J.gradient(g, x, None)
        res = J.checkGradient(x, g, 5, 1)
        errors = [l[-1] for l in res]
        self.assertTrue(errors[-1] < 0.11 * errors[-2])

    def test_fe(self):
        n = 100
        mesh = fd.UnitSquareMesh(n, n)

        inner = fs.LaplaceInnerProduct()
        Q = fs.FeControlSpace(mesh, inner)
        self.run_taylor_test(Q)

    def run_fe_mg(self, order):
        mesh = fd.UnitSquareMesh(10, 10)
        mesh = fd.Mesh(fd.Function(fd.VectorFunctionSpace(mesh, "CG", order)).interpolate(fd.SpatialCoordinate(mesh)))

        inner = fs.LaplaceInnerProduct()
        Q = fs.FeMultiGridControlSpace(mesh, inner, refinements_per_level=4)
        self.run_taylor_test(Q)

    def test_fe_mg_first_order(self):
        self.run_fe_mg(1)

    def test_fe_mg_second_order(self):
        self.run_fe_mg(2)


if __name__ == '__main__':
    unittest.main()
