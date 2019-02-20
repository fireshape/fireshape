import pytest
import firedrake as fd
import fireshape as fs
import fireshape.zoo as fsz
import ROL


def run_levelset_optimization(Q, inner, write_output=False):
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
    f = (pow(2*x, 2))+pow(y, 2) - 2.
    J = fsz.LevelsetFunctional(f, Q, cb=cb, scale=0.1)

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
            'Gradient Tolerance': 1e-6,
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
    assert (state.gnorm < 1e-6)

def run_levelset_optimization_3D(Q, inner, write_output=False):
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
    f = (pow(2*x, 2))+pow(1.5*y, 2)+pow(z, 2) - 2.
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
            'Gradient Tolerance': 1e-3,
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
    assert (state.gnorm < 1e-3)


def test_fe(pytestconfig):
    """Test for FeControlSpace with all inner products."""

    for inner in [fs.ElasticityInnerProduct,
                  fs.LaplaceInnerProduct,
                  fs.H1InnerProduct]:

        run_fe_test(inner)

def run_fe_test(inner, verbose=False):
    mesh = fs.DiskMesh(0.03)
    Q = fs.FeControlSpace(mesh)
    inner = inner(Q, direct_solve=True)
    run_levelset_optimization(Q, inner, write_output=verbose)

def test_fe_3D(pytestconfig):
    """3D Test for FeControlSpace."""
    mesh = fs.SphereMesh(0.1)
    Q = fs.FeControlSpace(mesh)
    inner = fs.LaplaceInnerProduct(Q)
    run_levelset_optimization_3D(Q, inner, write_output=pytestconfig.getoption("verbose"))

def run_fe_mg(order, write_output=False):
    """Test template for FeMultiGridControlSpace."""
    mesh = fs.DiskMesh(0.25)
    # State space mesh arises from 4 refinements of control space mesh
    Q = fs.FeMultiGridControlSpace(mesh, refinements=4,
                                   order=order)
    inner = fs.LaplaceInnerProduct(Q, direct_solve=True)
    run_levelset_optimization(Q, inner, write_output=write_output)

def run_fe_mg_3D(order, write_output=False):
    """Test template for FeMultiGridControlSpace."""
    mesh = fs.SphereMesh(0.2)
    # State space mesh arises from 4 refinements of control space mesh
    Q = fs.FeMultiGridControlSpace(mesh, refinements=1,
                                   order=order)
    inner = fs.LaplaceInnerProduct(Q)
    run_levelset_optimization_3D(Q, inner, write_output=write_output)

def test_fe_mg_first_order(pytestconfig):
    """Test FeMultiGridControlSpace with CG1 control."""
    run_fe_mg(1, write_output=pytestconfig.getoption("verbose"))
    run_fe_mg_3D(1, write_output=pytestconfig.getoption("verbose"))

def test_fe_mg_second_order(pytestconfig):
    """Test FeMultiGridControlSpace with CG2 control."""
    run_fe_mg(2, write_output=pytestconfig.getoption("verbose"))
    run_fe_mg_3D(2, write_output=pytestconfig.getoption("verbose"))

def test_bsplines(pytestconfig):
    """Test for BsplineControlSpace."""
    mesh = fs.DiskMesh(0.03)
    bbox = [(-3, 3), (-3,3)]
    orders = [3, 3]
    levels = [5, 5]
    Q = fs.BsplineControlSpace(mesh, bbox, orders, levels)
    inner = fs.H1InnerProduct(Q, direct_solve=True)
    run_levelset_optimization(Q, inner, write_output=pytestconfig.getoption("verbose"))

@pytest.mark.skip(reason="works locally, not on travis, have to figure out whats happening here at some point")
def test_bsplines_3D(pytestconfig):
    """3D Test for BsplineControlSpace."""
    mesh = fs.SphereMesh(0.1)
    bbox = [(-3, 3), (-3, 3), (-3,3)]
    orders = [2, 2, 2]
    levels =  [4, 4, 4]
    Q = fs.BsplineControlSpace(mesh, bbox, orders, levels)
    inner = fs.H1InnerProduct(Q)
    run_levelset_optimization_3D(Q, inner, write_output=pytestconfig.getoption("verbose"))


if __name__ == '__main__':
    pytest.main()
