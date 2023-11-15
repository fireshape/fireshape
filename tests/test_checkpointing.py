import pytest
import firedrake as fd
import fireshape as fs


@pytest.mark.parametrize("controlspace_t", [fs.FeControlSpace,
                                            fs.FeMultiGridControlSpace,
                                            fs.BsplineControlSpace])
def test_checkpointing(controlspace_t):
    mesh = fd.UnitSquareMesh(5, 5)

    if controlspace_t == fs.BsplineControlSpace:
        bbox = [(-1, 2), (-1, 2)]
        orders = [2, 2]
        levels = [4, 4]
        Q = fs.BsplineControlSpace(mesh, bbox, orders, levels)
    elif controlspace_t == fs.FeMultiGridControlSpace:
        Q = fs.FeMultiGridControlSpace(mesh, refinements=1, degree=2)
    else:
        Q = controlspace_t(mesh)

    inner = fs.H1InnerProduct(Q)

    q = fs.ControlVector(Q, inner)
    p = fs.ControlVector(Q, inner)

    from firedrake.petsc import PETSc
    rand = PETSc.Random().create(mesh.comm)
    rand.setInterval((1, 2))
    q.vec_wo().setRandom(rand)

    Q.store(q)

    Q.load(p)

    assert q.norm() > 0
    assert abs(q.norm()-p.norm()) < 1e-14
    p.axpy(-1, q)
    assert p.norm() < 1e-14
