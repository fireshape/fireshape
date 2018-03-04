import ROL
import firedrake as fd

__all__ = ["FeControlSpace", "FeMultiGridControlSpace",
           "BsplineControlSpace", "ControlVector"]

#new imports for splines
from firedrake.petsc import PETSc
from functools import reduce
from scipy.interpolate import splev
import numpy as np

class ControlSpace(object):
    """
    ControlSpace is the space of geometric transformations.

    A transformation is identified with a domain using Firedrake.
    In particular, a transformation is converted into a domain by
    interpolating it on a Firedrake Lagrangian finite element space.

    Notational convention:
        self.mesh_r is the initial physical mesh (reference domain)
        self.V_r is the Firedrake vectorial Lagrangian finite element
            space on mesh_r
        self.id is the element of V_r that satisfies id(x) = x for every x
        self.T is the interpolant of a ControlSpace variable in self.V_r
        self.mesh_m is the mesh that corresponds to self.T (moved domain)
        self.V_m is the Firedrake vectorial Lagrangian finite element
            space on mesh_m
        self.inner_product is the inner product of the ControlSpace
    """
    def restrict(self, residual, out):
        """
        Restrict from self.V_r into ControlSpace

        Input:
        residual: fd.Function, is a variable in the dual of self.V_r
        out: ControlVector, is a variable in the dual of ControlSpace
             (overwritten with result)
        """

        raise NotImplementedError

    def interpolate(self, vector, out):
        """
        Interpolate from ControlSpace into self.V_r

        Input:
        vector: ControlVector, is a variable in ControlSpace
        out: fd.Function, is a variable in self.V_r, is overwritten with
             the result
        """

        raise NotImplementedError

    def update_domain(self, q: 'ControlVector'):
        """
        Update the interpolant self.T with q
        """

        self.interpolate(q, self.T)
        self.T += self.id

    def get_zero_vec(self):
        """
        Create the object that stores the data for a ControlVector.

        It returns a fd.Function or a PETSc.Vec.
        It is only used in ControlVector.__init__.
        """

        raise NotImplementedError

    def assign_inner_product(self, inner_product):
        """
        create self.inner_product
        """
        raise NotImplementedError


class FeControlSpace(ControlSpace):

    def __init__(self, mesh_r, inner_product):
        self.mesh_r = mesh_r
        element = self.mesh_r.coordinates.function_space().ufl_element()
        self.V_r = fd.FunctionSpace(self.mesh_r, element)
        X = fd.SpatialCoordinate(self.mesh_r)
        self.id = fd.interpolate(X, self.V_r)
        self.T = fd.Function(self.V_r, name="T")
        self.T.assign(self.id)
        self.mesh_m = fd.Mesh(self.T)
        self.V_m = fd.FunctionSpace(self.mesh_m, element)
        self.inner_product = inner_product
        self.inner_product.get_impl(self.V_r)

    def restrict(self, residual, out):
        with residual.dat.vec as vecres:
            vecres.copy(out.vec)

    def interpolate(self, vector, out):
        out.assign(vector.fun)

    def get_zero_vec(self):
        fun = fd.Function(self.V_r)
        fun *= 0.
        return fun


class FeMultiGridControlSpace(ControlSpace):

    def __init__(self, mesh_r, inner_product, refinements=1, order=1):
        mh = fd.MeshHierarchy(mesh_r, 1, refinements_per_level=refinements)
        self.mesh_hierarchy = mh
        self.mesh_r_coarse = self.mesh_hierarchy[0]
        self.V_r_coarse = fd.VectorFunctionSpace(self.mesh_r_coarse, "CG",
                                                 order)

        self.mesh_r = self.mesh_hierarchy[1]
        element = self.V_r_coarse.ufl_element()
        self.V_r = fd.FunctionSpace(self.mesh_r, element)
        X = fd.SpatialCoordinate(self.mesh_r)
        self.id = fd.Function(self.V_r).interpolate(X)
        self.T = fd.Function(self.V_r, name="T")
        self.T.assign(self.id)
        self.mesh_m = fd.Mesh(self.T)
        self.V_m = fd.FunctionSpace(self.mesh_m, element)

        self.inner_product = inner_product
        self.inner_product.get_impl(self.V_r_coarse)

    def restrict(self, residual, out):
        fd.restrict(residual, out.fun)

    def interpolate(self, vector, out):
        fd.prolong(vector.fun, out)

    def get_zero_vec(self):
        fun = fd.Function(self.V_r_coarse)
        fun *= 0.
        return fun


class BsplineControlSpace(ControlSpace):
    """ConstrolSpace based on cartesian tensorized Bsplines."""
    def __init__(self, mesh, inner_product, bbox, orders, levels):
        """
        bbox: a list of tuples describing [(xmin, xmax), (ymin, ymax), ...]
              of a Cartesian grid that extends around the shape to be
              optimised
        orders: describe the orders (one integer per geometric dimension)
                of the tensor-product B-spline basis. A univariate B-spline
                has order "o" if it is a piecewise polynomial of degree
                "o-1". For instance, a hat function is a B-spline of
                order 2 and thus degree 1.
        levels: describe the subdivision levels (one integers per
                geometric dimension) used to construct the knots of
                univariate B-splines
        """
        # information on B-splines
        self.dim = len(bbox) # geometric dimension
        self.bbox = bbox
        self.orders = orders
        self.levels = levels
        self.construct_knots()

        # create temporary self.mesh_r and self.V_r to assemble inner product
        if self.dim == 2:
            nx = len(self.knots[0]) - 1
            ny = len(self.knots[1]) - 1
            Lx = self.bbox[0][1] - self.bbox[0][0]
            Ly = self.bbox[1][1] - self.bbox[1][0]
            meshloc = fd.RectangleMesh(nx, ny, Lx, Ly, quadrilateral=True,
                                    comm = mesh.mpi_comm()) #quadrilaterals or triangle?
            # shift in x- and y-direction
            meshloc.coordinates.dat.data[:,0] += self.bbox[0][0]
            meshloc.coordinates.dat.data[:,1] += self.bbox[1][0]
            inner_product.fixed_bids = [1,2,3,4]

        elif self.dim == 3:
            # maybe use extruded meshes, quadrilateral not available
            nx = len(self.knots[0]) - 1
            ny = len(self.knots[1]) - 1
            nz = len(self.knots[2]) - 1
            Lx = self.bbox[0][1] - self.bbox[0][0]
            Ly = self.bbox[1][1] - self.bbox[1][0]
            Ly = self.bbox[2][1] - self.bbox[2][0]
            meshloc = BoxMesh(nx, ny, nz, Lx, Ly, Lz,
                              comm = mesh_r.mpi_comm())
            # shift in x-, y-, and z-direction
            meshloc.coordinates.dat.data[:,0] += self.bbox[0][0]
            meshloc.coordinates.dat.data[:,1] += self.bbox[1][0]
            meshloc.coordinates.dat.data[:,2] += self.bbox[2][0]
            inner_product.fixed_bids = [1,2,3,4,5,6]

        self.mesh_r = meshloc
        maxdegree = max(self.orders)-1
        self.V_r = fd.VectorFunctionSpace(self.mesh_r, "CG", maxdegree ) #is this the proper space?
        self.build_interpolation_matrix()
        self.inner_product = inner_product
        self.inner_product.get_impl(self.V_r, self.FullIFW)

        # standard construction of ControlSpace
        self.mesh_r = mesh
        element = self.mesh_r.coordinates.function_space().ufl_element()
        self.V_r = fd.FunctionSpace(self.mesh_r, element)
        X = fd.SpatialCoordinate(self.mesh_r)
        self.id = fd.Function(self.V_r).interpolate(X)
        self.T = fd.Function(self.V_r, name="T")
        self.T.assign(self.id)
        self.mesh_m = fd.Mesh(self.T)
        self.V_m = fd.FunctionSpace(self.mesh_m, element)

        assert self.dim == self.mesh_r.geometric_dimension()

        # assemble correct interpolation matrix
        self.build_interpolation_matrix()


    def construct_knots(self):
        """
        construct self.knots, self.n, self.N

        self.knots is a list of np.arrays (one per geometric dimension)
        each array corresponds to the knots used to define the spline space

        self.n is a list of univariate spline space dimensions
            (one per geometric dim)

        self.N is the dimension of the scalar tensorized spline space
        """
        self.knots = []
        self.n = []
        for dim in range(self.dim):
            order = self.orders[dim]
            level = self.levels[dim]

            assert order >= 1
            #degree = order-1 # splev uses degree, not order
            assert level >= 1 # with level=1 only bdry Bsplines

            knots_01 = np.concatenate((np.zeros((order-1,), dtype=float),
                                       np.linspace(0., 1., 2**level+1),
                                       np.ones((order-1,), dtype=float)))

            (xmin, xmax) = self.bbox[dim]
            knots = (xmax - xmin)*knots_01 + xmin
            self.knots.append(knots)
            # dimension of univariate spline spaces
            # the "-2" is because we want homogeneous Dir bc
            n = len(knots) - order - 2
            assert n > 0
            self.n.append(n)

        # dimension of multivariate spline space
        N = reduce(lambda x, y: x*y, self.n)
        self.N = N

    def build_interpolation_matrix(self):
        """
        Construct the matrix self.FullIFW.

        The columns of self.FullIFW are the interpolant
        of (vectorial tensorized) Bsplines into self.V_r
        """
        # construct list of scalar univariate interpolation matrices
        interp_1d = self.construct_1d_interpolation_matrices()
        # construct scalar tensorial interpolation matrix
        IFW = self.construct_kronecker_matrix(interp_1d)
        # interleave self.dim-many IFW matrices among each other
        self.FullIFW = self.construct_full_interpolation_matrix(IFW)

    def construct_1d_interpolation_matrices(self):
        """
        Create a list of sparse matrices (one per geometric dimension).

        Each matrix has size (M, n[dim]), where M is the dimension of the
        self.V_r.sub(0), and n[dim] is the dimension of the univariate
        spline space associated to the dimth-geometric coordinate.
        The ith column of such a matrix is computed by evaluating the ith
        univariate B-spline on the dimth-geometric coordinate of the dofs
        of self.V_r(0)
        """
        interp_1d = []

        # this code is correct but can be made more beautiful
        # by replacing x_fct with self.id
        x_fct = fd.SpatialCoordinate(self.mesh_r) #used for x_int
        # compute self.M, x_int will be overwritten below
        x_int = fd.interpolate(x_fct[0], self.V_r.sub(0))
        self.M = x_int.vector().size()

        for dim in range(self.dim):
            order = self.orders[dim]
            knots = self.knots[dim]
            n = self.n[dim]

            I = PETSc.Mat().create(comm=self.mesh_r.mpi_comm())
            I.setType(PETSc.Mat.Type.AIJ)
            I.setSizes((self.M, n))
            # BIG TODO: figure out the sparsity pattern
            I.setUp()

            x_int = fd.interpolate(x_fct[dim], self.V_r.sub(0))
            with x_int.dat.vec_ro as x:
                for idx in range(n):
                    coeffs = np.zeros(knots.shape, dtype=float)
                    coeffs[idx+1] = 1  # idx+1 because we impose hom Dir bc
                    degree = order - 1 # splev uses degree, not order
                    tck = (knots, coeffs, degree)

                    values = splev(x.array, tck, der=0, ext=1)
                    rows = np.where(values != 0)[0].astype(np.int32)
                    values = values[rows]
                    I.setValues(rows, [idx], values)

            I.assemble() # lazy strategy for kron
            interp_1d.append(I)

        return interp_1d

    def construct_kronecker_matrix(self, interp_1d):
        """
        Construct the tensorized interpolation matrix.

        Do this by computing the kron product of the rows of
        the 1d univariate interpolation matrices.
        In the future, this may be done matrix-free.
        """
        IFW = PETSc.Mat().create(self.mesh_r.mpi_comm())
        IFW.setType(PETSc.Mat.Type.AIJ)
        IFW.setSizes((self.M, self.N))
        # BIG TODO: figure out the sparsity pattern
        IFW.setUp()

        for row in range(self.M):
            rows = [A.getRow(row) for A in interp_1d]
            denserows = [np.zeros((n,)) for n in self.n]
            for ii in range(self.dim):
                denserows[ii][rows[ii][0]] = rows[ii][1]

            values = reduce(np.kron, denserows)
            columns = np.where(values != 0)[0].astype(np.int32)
            values = values[columns]
            IFW.setValues([row], columns, values)

        IFW.assemble()
        return IFW

    def construct_full_interpolation_matrix(self, IFW):
        """
        Assemble interpolation matrix for vectorial tensorized spline space.
        """
        FullIFW = PETSc.Mat().create(self.mesh_r.mpi_comm())
        FullIFW.setType(PETSc.Mat.Type.AIJ)
        FullIFW.setSizes((self.dim * self.M, self.dim * self.N))
        # BIG TODO: figure out the sparsity pattern
        FullIFW.setUp()

        # this blows up the matrix to do the right thing
        # on vector fields. It's not just a block matrix,
        # but the values are interleaved as this is how
        # firedrake handles vector fields
        for row in range(self.M):
            (cols, vals) = IFW.getRow(row)
            for dim in range(self.dim):
                FullIFW.setValues([self.dim * row + dim],
                                  [self.dim * col + dim for col in cols],
                                  vals)
        FullIFW.assemble()
        return FullIFW

    def restrict(self, residual, out):
        with residual.dat.vec as w:
            self.FullIFW.multTranspose(w, out.vec)

    def interpolate(self, vector, out):
        with out.dat.vec as w:
            self.FullIFW.mult(vector.vec, w)

    def get_zero_vec(self):
        vec = PETSc.Vec().createSeq(self.N*self.dim,
                                    comm=self.mesh_r.mpi_comm())
        return vec

class ControlVector(ROL.Vector):
    """
    A ControlVector is a variable in the ControlSpace.

    The data of a control vector is a PETSc.vec stored in self.vec.
    If this data corresponds also to a fd.Function, the firedrake wrapper
    around self.vec is stored in self.fun (otherwise, self.fun = None).

    A ControlVector is a ROL.Vector and thus needs the following methods:
    plus, scale, clone, dot, axpy, set.
    """
    def __init__(self, controlspace: ControlSpace, data=None):
        super().__init__()
        self.controlspace = controlspace

        if data is None:
            data = controlspace.get_zero_vec()

        if isinstance(data, fd.Function):
            self.fun = data
            with data.dat.vec as v:
                self.vec = v
        else:
            self.vec = data
            self.fun = None

    def plus(self, v):
        self.vec += v.vec

    def scale(self, alpha):
        self.vec *= alpha

    def clone(self):
        """
        Returns a zero vector of the same size of self.

        The name of this method is misleading, but it is dictated by ROL.
        """
        res = ControlVector(self.controlspace)
        # res.set(self)
        return res

    def dot(self, v):
        """Inner product between self and v."""
        return self.controlspace.inner_product.eval(self, v)

    def axpy(self, alpha, x):
        self.vec.axpy(alpha, x.vec)

    def set(self, v):
        v.vec.copy(self.vec)

    def __str__(self):
        """String representative, so we can call print(vec)."""
        return self.vec[:].__str__()
