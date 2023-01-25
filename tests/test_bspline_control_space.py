import pytest
import firedrake as fd
import fireshape as fs


@pytest.mark.parametrize("dim", [2, 3])
@pytest.mark.parametrize("inner_t", [fs.H1InnerProduct,
                                     fs.ElasticityInnerProduct,
                                     fs.LaplaceInnerProduct])
@pytest.mark.parametrize("order", [2, 3])
def test_bspline_control_space(dim, inner_t, order, pytestconfig):
    """ Test template for fs.BsplineControlSpace."""

    if dim == 2:
        mesh = fs.DiskMesh(0.1)
        bbox = [(-2, 2), (-2, 2)]
        orders = [order] * 2
        levels = [4, 4]
    elif dim == 3:
        mesh = fs.SphereMesh(0.2)
        bbox = [(-3, 3), (-3, 3), (-3, 3)]
        orders = [order] * 3
        levels = [2, 2, 2]
    else:
        raise NotImplementedError

    Q = fs.BsplineControlSpace(mesh, bbox, orders, levels)
    A = inner_t(Q).A

    V = Q.V_control
    meshloc = V.mesh()
    elem = V.ufl_element()
    degree = elem.degree()
    Q.mesh_r = meshloc

    if degree > 1:
        Q.V_control = fd.FunctionSpace(
            meshloc, elem.reconstruct(degree=degree-1))
        Q.I_control = Q.build_interpolation_matrix(Q.V_control)
        A_lower_order = inner_t(Q).A
        assert (A_lower_order - A).norm() > 1e-12

    Q.V_control = fd.FunctionSpace(
        meshloc, elem.reconstruct(degree=degree+1))
    Q.I_control = Q.build_interpolation_matrix(Q.V_control)
    A_higher_order = inner_t(Q).A
    assert (A_higher_order - A).norm() < 1e-12


if __name__ == '__main__':
    pytest.main()
