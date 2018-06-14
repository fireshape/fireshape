import pytest
import firedrake as fd
import fireshape as fs
import fireshape.zoo as fsz


def run_taylor_test(Q):
    f = fd.Constant(1.0)

    x = fs.ControlVector(Q)
    J = fsz.LevelsetFunctional(f, Q)
    """
    move mesh a bit to check that we are not doing the
    taylor test in T=id
    """
    g = x.clone()
    J.gradient(g, x, None)
    x.plus(g)
    J.update(x, None, 1)

    """ Start taylor test """
    J.gradient(g, x, None)
    res = J.checkGradient(x, g, 5, 1)
    errors = [l[-1] for l in res]
    assert (errors[-1] < 0.11 * errors[-2])

def test_fe(self):
    n = 100
    mesh = fd.UnitSquareMesh(n, n)

    inner = fs.LaplaceInnerProduct()
    Q = fs.FeControlSpace(mesh, inner)
    run_taylor_test(Q)

def run_fe_mg(order):
    mesh = fd.UnitSquareMesh(10, 10)

    inner = fs.H1InnerProduct()
    Q = fs.FeMultiGridControlSpace(mesh, inner, refinements=4, order=order)
    run_taylor_test(Q)

def test_fe_mg_first_order():
    run_fe_mg(1)

def test_fe_mg_second_order():
    run_fe_mg(2)


if __name__ == '__main__':
    pytest.main()
