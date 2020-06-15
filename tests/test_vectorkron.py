import pytest
import firedrake as fd
import fireshape as fs
import scipy as sp
import numpy as np


def mytest_vectorkron(Q, A, B):
    """ Test template for fs.BsplineControlSpace.vectorkron."""

    v = [A.indices, A.data, A.shape[1]]
    w = [B.indices, B.data, B.shape[1]]
    kvw = Q.vectorkron(v, w)
    kAB = sp.sparse.kron(A, B, format="csr")
    if len(kAB.data) == 0:
        kAB.indices = np.array([])
    same_ind = sum(abs(kvw[0] - kAB.indices)) < 1e-10
    same_dat = sum(abs(kvw[1] - kAB.data)) < 1e-10
    assert same_ind & same_dat


def test_vectorkron():
    """ Collection of tests."""

    mesh = fd.UnitSquareMesh(1, 1)
    bbox = [(0, 1), (0, 1)]
    orders = [2, 2]
    levels = [2, 2]
    Q = fs.BsplineControlSpace(mesh, bbox, orders, levels)

    # test
    A = sp.sparse.csr_matrix(np.array([[0, 5, 7, 0, 9]]))
    B = sp.sparse.csr_matrix(np.array([[1, 2, 0]]))
    mytest_vectorkron(Q, A, B)

    # test
    A = sp.sparse.csr_matrix(np.array([[0, 2, 0, 5, 6]]))
    B = sp.sparse.csr_matrix(np.array([[0, 2, 0, 5, 6]]))
    mytest_vectorkron(Q, A, B)

    # test
    A = sp.sparse.csr_matrix(np.array([[]]))
    B = sp.sparse.csr_matrix(np.array([[0, 2, 0, 5, 6]]))
    mytest_vectorkron(Q, A, B)

    # test
    A = sp.sparse.csr_matrix(np.array([[1, 1.1, 10, 0, 2, 0, 5, 6]]))
    B = sp.sparse.csr_matrix(np.array([[0, 2, 0, -3, 6]]))
    mytest_vectorkron(Q, A, B)

    # test
    A = sp.sparse.csr_matrix(np.array([[0, 2, 0, 5, 6]]))
    B = sp.sparse.csr_matrix(np.array([[]]))
    mytest_vectorkron(Q, A, B)

    # test
    A = sp.sparse.csr_matrix(np.array([[0, 5, 6]]))
    B = sp.sparse.csr_matrix(np.array([[1e3, 0, 2, 0, 5, 6]]))
    mytest_vectorkron(Q, A, B)


if __name__ == '__main__':
    pytest.main()
