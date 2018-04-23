import unittest
import firedrake as fd
import fireshape as fs
import fireshape.zoo as fsz
import ROL

class LevelsetTest(unittest.TestCase):

    def run_levelset_optimization(self, Q, inner, write_output=False):
        """ Test template for fsz.LevelsetFunctional."""

        #tool for developing new tests, allows storing shape iterates
        if write_output:
            out = fd.File("domain.pvd")

            def cb(*args):
                out.write(Q.mesh_m.coordinates)

            cb()
        else:
            cb = None

        # levelset test case
        (x, y) = fd.SpatialCoordinate(Q.mesh_m)
        f = (pow(x-0.5, 2))+pow(y-0.5, 2) - 2.
        J = fsz.LevelsetFunctional(f, Q, cb=cb)

        q = fs.ControlVector(Q, inner)

        # ROL parameters
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

        # assemble and solve ROL optimization problem
        params = ROL.ParameterList(params_dict, "Parameters")
        problem = ROL.OptimizationProblem(J, q)
        solver = ROL.OptimizationSolver(problem, params)
        solver.solve()

        # verify that the norm of the gradient at optimum is small enough
        state = solver.getAlgorithmState()
        self.assertTrue(state.gnorm < 1e-6)

    def run_levelset_optimization_3D(self, Q, inner, write_output=False):
        """ Test template for fsz.LevelsetFunctional."""

        #tool for developing new tests, allows storing shape iterates
        if write_output:
            out = fd.File("domain.pvd")

            def cb(*args):
                out.write(Q.mesh_m.coordinates)

            cb()
        else:
            cb = None

        # levelset test case
        (x, y, z) = fd.SpatialCoordinate(Q.mesh_m)
        f = (pow(x-0.5, 2))+pow(y-0.5, 2)+pow(z-0.5, 2) - 2.
        J = fsz.LevelsetFunctional(f, Q, cb=cb)
        q = fs.ControlVector(Q, inner)

        # ROL parameters
        params_dict = {
            'General': {
                'Secant': {'Type': 'Limited-Memory BFGS',
                           'Maximum Storage': 25}},
            'Step': {
                'Type': 'Line Search',
                'Line Search': {'Descent Method': {
                    'Type': 'Quasi-Newton Step'}}},
            'Status Test': {
                'Gradient Tolerance': 1e-4,
                'Relative Gradient Tolerance': 1e-3,
                'Step Tolerance': 1e-10, 'Relative Step Tolerance': 1e-10,
                'Iteration Limit': 50}}

        # assemble and solve ROL optimization problem
        params = ROL.ParameterList(params_dict, "Parameters")
        problem = ROL.OptimizationProblem(J, q)
        solver = ROL.OptimizationSolver(problem, params)
        solver.solve()

        # verify that the norm of the gradient at optimum is small enough
        state = solver.getAlgorithmState()
        self.assertTrue(state.gnorm < 1e-4)


    def test_fe(self):
        """Test for FeControlSpace with all inner products."""

        for inner in [fs.ElasticityInnerProduct,
                      fs.LaplaceInnerProduct,
                      fs.H1InnerProduct]:
            
            self.run_fe_test(inner)

    def run_fe_test(self, inner):
        n = 100
        mesh = fd.UnitSquareMesh(n, n)
        Q = fs.FeControlSpace(mesh)
        inner = inner(Q)
        self.run_levelset_optimization(Q, inner, write_output=False)

    def test_fe_3D(self):
        """3D Test for FeControlSpace."""
        n = 30
        mesh = fd.UnitCubeMesh(n, n, n)
        Q = fs.FeControlSpace(mesh)
        inner = fs.LaplaceInnerProduct(Q)
        self.run_levelset_optimization_3D(Q, inner, write_output=False)

    def run_fe_mg(self, order, write_output=False):
        """Test template for FeMultiGridControlSpace."""
        mesh = fd.UnitSquareMesh(4, 4)
        # State space mesh arises from 4 refinements of control space mesh
        Q = fs.FeMultiGridControlSpace(mesh, refinements=4,
                                       order=order)
        inner = fs.LaplaceInnerProduct(Q)
        self.run_levelset_optimization(Q, inner, write_output=write_output)

    #currently not supported by firedrake
    #def run_fe_mg_3D(self, order, write_output=False):
    #    """3D Test template for FeMultiGridControlSpace."""
    #    mesh = fd.UnitCubeMesh(4, 4, 4)
    #    inner = fs.LaplaceInnerProduct()
    #    # State space mesh arises from 4 refinements of control space mesh
    #    Q = fs.FeMultiGridControlSpace(mesh, inner, refinements=4,
    #                                   order=order)
    #    self.run_levelset_optimization_3D(Q, write_output=write_output)

    def test_fe_mg_first_order(self):
        """Test FeMultiGridControlSpace with CG1 control."""
        self.run_fe_mg(1, write_output=False)
        #self.run_fe_mg_3D(1, write_output=False)

    def test_fe_mg_second_order(self):
        """Test FeMultiGridControlSpace with CG2 control."""
        self.run_fe_mg(2, write_output=False)
        #self.run_fe_mg_3D(2, write_output=False)

    def run_fe_mg_3D(self, order, write_output=False):
        """Test template for FeMultiGridControlSpace."""
        mesh = fd.UnitCubeMesh(4, 4, 4)
        # State space mesh arises from 4 refinements of control space mesh
        Q = fs.FeMultiGridControlSpace(mesh, refinements=4,
                                       order=order)
        inner = fs.LaplaceInnerProduct(Q)
        self.run_levelset_optimization_3D(Q, inner, write_output=write_output)

    def test_bsplines(self):
        """Test for BsplineControlSpace."""
        n = 100
        mesh = fd.UnitSquareMesh(n, n)
        bbox = [(-3, 4), (-3,4)]
        orders = [3, 3]
        levels = [5, 5]
        Q = fs.BsplineControlSpace(mesh, bbox, orders, levels)
        inner = fs.H1InnerProduct(Q)
        self.run_levelset_optimization(Q, inner, write_output=False)

    def test_bsplines_3D(self):
        """3D Test for BsplineControlSpace."""
        n = 20
        mesh = fd.UnitCubeMesh(n, n, n)
        bbox = [(-3, 4), (-3, 4), (-3,4)]
        orders = [2, 2, 2]
        levels =  [3, 3, 3]
        Q = fs.BsplineControlSpace(mesh, bbox, orders, levels)
        inner = fs.H1InnerProduct(Q)
        self.run_levelset_optimization_3D(Q, inner, write_output=True)


if __name__ == '__main__':
    unittest.main()
