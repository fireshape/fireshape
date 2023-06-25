from .innerproduct import InnerProduct
import ROL
import firedrake as fd

__all__ = ["FeControlSpace", "FeMultiGridControlSpace", "BsplineControlSpace",
           "WaveletControlSpace", "ControlVector"]

# new imports for splines
from firedrake.petsc import PETSc
from functools import reduce
from scipy.interpolate import splev
import numpy as np

# new imports for wavelets
from itertools import product
from math import factorial, floor, ceil
from scipy.special import binom


class ControlSpace(object):
    """
    ControlSpace is the space of geometric transformations.

    A transformation is identified with a domain using Firedrake.
    In particular, a transformation is converted into a domain by
    interpolating it on a Firedrake Lagrangian finite element space.

    Notational convention:
        self.mesh_r is the initial physical mesh (reference domain)
        self.V_r is a Firedrake vectorial Lagrangian finite element
            space on mesh_r
        self.id is the element of V_r that satisfies id(x) = x for every x
        self.T is the interpolant of a ControlSpace variable in self.V_r
        self.mesh_m is the mesh that corresponds to self.T (moved domain)
        self.V_m is the Firedrake vectorial Lagrangian finite element
            space on mesh_m
        self.inner_product is the inner product of the ControlSpace

    Key idea: solve state and adjoint equations on mesh_m. Then, evaluate
    shape derivatives along directions in V_m, transplant this to V_r,
    restrict to ControlSpace, compute the update (using inner_product),
    and interpolate it into V_r to update mesh_m.

    Note: transplant to V_r means creating an function in V_r and using the
    values of the directional shape derivative as coefficients. Since
    Lagrangian finite elements are parametric, no transformation matrix is
    required by this operation.
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

        # Check if the new control is different from the last one.  ROL is
        # sometimes a bit strange in that it calls update on the same value
        # more than once, in that case we don't want to solve the PDE over
        # again.

        if not hasattr(self, 'lastq') or self.lastq is None:
            self.lastq = q.clone()
            self.lastq.set(q)
        else:
            self.lastq.axpy(-1., q)
            # calculate l2 norm (faster)
            diff = self.lastq.vec_ro().norm()
            self.lastq.axpy(+1., q)
            if diff < 1e-20:
                return False
            else:
                self.lastq.set(q)
        q.to_coordinatefield(self.T)
        self.T += self.id
        return True

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

    def get_space_for_inner(self):
        """
        Return the functionspace V to define the inner product on
        and possibly an interpolation matrix I between the finite element
        functions in V and the control functions. Note that this matrix
        is not necessarily related to self.restict() and self.interpolate()
        """
        raise NotImplementedError

    def store(self, vec, filename):
        """
        Store the vector to a file to be reused in a later computation
        """
        raise NotImplementedError

    def load(self, vec, filename):
        """
        Load a vector from a file
        """
        raise NotImplementedError


class FeControlSpace(ControlSpace):
    """Use self.V_r as actual ControlSpace."""

    def __init__(self, mesh_r):
        # Create mesh_r and V_r
        self.mesh_r = mesh_r
        element = self.mesh_r.coordinates.function_space().ufl_element()
        self.V_r = fd.FunctionSpace(self.mesh_r, element)

        # Create self.id and self.T, self.mesh_m, and self.V_m.
        X = fd.SpatialCoordinate(self.mesh_r)
        self.id = fd.interpolate(X, self.V_r)
        self.T = fd.Function(self.V_r, name="T")
        self.T.assign(self.id)
        self.mesh_m = fd.Mesh(self.T)
        self.V_m = fd.FunctionSpace(self.mesh_m, element)
        self.is_DG = False

        """
        ControlSpace for discontinuous coordinate fields
        (e.g.  periodic domains)

        In Firedrake, periodic meshes are implemented using a discontinuous
        field. This implies that self.V_r contains discontinuous functions.
        To ensure domain updates do not create holes in the domain,
        use a continuous subspace self.V_c of self.V_r as control space.
        """
        if element.family() == 'Discontinuous Lagrange':
            self.is_DG = True
            self.V_c = fd.VectorFunctionSpace(self.mesh_r,
                                              "CG", element._degree)
            self.Ip = fd.Interpolator(fd.TestFunction(self.V_c),
                                      self.V_r).callable().handle

    def restrict(self, residual, out):
        if self.is_DG:
            with residual.dat.vec as w:
                self.Ip.multTranspose(w, out.vec_wo())
        else:
            with residual.dat.vec as vecres:
                with out.fun.dat.vec as vecout:
                    vecres.copy(vecout)

    def interpolate(self, vector, out):
        if self.is_DG:
            with out.dat.vec as w:
                self.Ip.mult(vector.vec_ro(), w)
        else:
            out.assign(vector.fun)

    def get_zero_vec(self):
        if self.is_DG:
            fun = fd.Function(self.V_c)
        else:
            fun = fd.Function(self.V_r)
        fun *= 0.
        return fun

    def get_space_for_inner(self):
        if self.is_DG:
            return (self.V_c, None)
        return (self.V_r, None)

    def store(self, vec, filename="control"):
        """
        Store the vector to a file to be reused in a later computation.
        DumbCheckpoint requires that the mesh, FunctionSpace and parallel
        decomposition are identical between store and load.

        """
        with fd.DumbCheckpoint(filename, mode=fd.FILE_CREATE) as chk:
            chk.store(vec.fun, name=filename)

    def load(self, vec, filename="control"):
        """
        Load a vector from a file.
        DumbCheckpoint requires that the mesh, FunctionSpace and parallel
        decomposition are identical between store and load.
        """
        with fd.DumbCheckpoint(filename, mode=fd.FILE_READ) as chk:
            chk.load(vec.fun, name=filename)


class FeMultiGridControlSpace(ControlSpace):
    """
    FEControlSpace on given mesh and StateSpace on uniformly refined mesh.

    Use the provided mesh to construct a Lagrangian finite element control
    space. Then, refine the mesh `refinements`-times to construct
    representatives of ControlVectors that are compatible with the state
    space.

    Inputs:
        refinements: type int, number of uniform refinements to perform
                     to obtain the StateSpace mesh.
        order: type int, order of Lagrange basis functions of ControlSpace.

    Note: as of 04.03.2018, 3D is not supported by fd.MeshHierarchy.
    """

    def __init__(self, mesh_r, refinements=1, order=1):
        mh = fd.MeshHierarchy(mesh_r, refinements)
        self.mesh_hierarchy = mh

        # Control space on coarsest mesh
        self.mesh_r_coarse = self.mesh_hierarchy[0]
        self.V_r_coarse = fd.VectorFunctionSpace(self.mesh_r_coarse, "CG",
                                                 order)

        # Create self.id and self.T on refined mesh.
        element = self.V_r_coarse.ufl_element()

        self.intermediate_Ts = []
        for i in range(refinements - 1):
            mesh = self.mesh_hierarchy[i + 1]
            V = fd.FunctionSpace(mesh, element)
            self.intermediate_Ts.append(fd.Function(V))

        self.mesh_r = self.mesh_hierarchy[-1]
        element = self.V_r_coarse.ufl_element()
        self.V_r = fd.FunctionSpace(self.mesh_r, element)

        X = fd.SpatialCoordinate(self.mesh_r)
        self.id = fd.Function(self.V_r).interpolate(X)
        self.T = fd.Function(self.V_r, name="T")
        self.T.assign(self.id)
        self.mesh_m = fd.Mesh(self.T)
        self.V_m = fd.FunctionSpace(self.mesh_m, element)

    def restrict(self, residual, out):
        Tf = residual
        for Tinter in reversed(self.intermediate_Ts):
            fd.restrict(Tf, Tinter)
            Tf = Tinter
        fd.restrict(Tf, out.fun)

    def interpolate(self, vector, out):
        Tc = vector.fun
        for Tinter in self.intermediate_Ts:
            fd.prolong(Tc, Tinter)
            Tc = Tinter
        fd.prolong(Tc, out)

    def get_zero_vec(self):
        fun = fd.Function(self.V_r_coarse)
        fun *= 0.
        return fun

    def get_space_for_inner(self):
        return (self.V_r_coarse, None)

    def store(self, vec, filename="control"):
        """
        Store the vector to a file to be reused in a later computation.
        DumbCheckpoint requires that the mesh, FunctionSpace and parallel
        decomposition are identical between store and load.

        """
        with fd.DumbCheckpoint(filename, mode=fd.FILE_CREATE) as chk:
            chk.store(vec.fun, name=filename)

    def load(self, vec, filename="control"):
        """
        Load a vector from a file.
        DumbCheckpoint requires that the mesh, FunctionSpace and parallel
        decomposition are identical between store and load.
        """
        with fd.DumbCheckpoint(filename, mode=fd.FILE_READ) as chk:
            chk.load(vec.fun, name=filename)


class BsplineControlSpace(ControlSpace):
    """ConstrolSpace based on cartesian tensorized Bsplines."""

    def __init__(self, mesh, bbox, orders, levels, fixed_dims=[],
                 boundary_regularities=None):
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
        fixed_dims: dimensions in which the deformation should be zero

        boundary_regularities: how fast the splines go to zero on the boundary
                               for each dimension
                               [0,..,0] : they don't go to zero
                               [1,..,1] : they go to zero with C^0 regularity
                               [2,..,2] : they go to zero with C^1 regularity
        """
        self.boundary_regularities = [o - 1 for o in orders] \
            if boundary_regularities is None else boundary_regularities
        # information on B-splines
        self.dim = len(bbox)  # geometric dimension
        self.bbox = bbox
        self.orders = orders
        self.levels = levels
        if isinstance(fixed_dims, int):
            fixed_dims = [fixed_dims]
        self.fixed_dims = fixed_dims
        self.construct_knots()
        self.comm = mesh.mpi_comm()
        # create temporary self.mesh_r and self.V_r to assemble innerproduct
        if self.dim == 2:
            nx = 2**levels[0]
            ny = 2**levels[1]
            Lx = self.bbox[0][1] - self.bbox[0][0]
            Ly = self.bbox[1][1] - self.bbox[1][0]
            meshloc = fd.RectangleMesh(nx, ny, Lx, Ly, quadrilateral=True,
                                       comm=self.comm)  # quads or triangle?
            # shift in x- and y-direction
            meshloc.coordinates.dat.data[:, 0] += self.bbox[0][0]
            meshloc.coordinates.dat.data[:, 1] += self.bbox[1][0]
            # inner_product.fixed_bids = [1,2,3,4]

        elif self.dim == 3:
            # maybe use extruded meshes, quadrilateral not available
            nx = 2**levels[0]
            ny = 2**levels[1]
            nz = 2**levels[2]
            Lx = self.bbox[0][1] - self.bbox[0][0]
            Ly = self.bbox[1][1] - self.bbox[1][0]
            Lz = self.bbox[2][1] - self.bbox[2][0]
            meshloc = fd.BoxMesh(nx, ny, nz, Lx, Ly, Lz, comm=self.comm)
            # shift in x-, y-, and z-direction
            meshloc.coordinates.dat.data[:, 0] += self.bbox[0][0]
            meshloc.coordinates.dat.data[:, 1] += self.bbox[1][0]
            meshloc.coordinates.dat.data[:, 2] += self.bbox[2][0]
            # inner_product.fixed_bids = [1,2,3,4,5,6]

        self.mesh_r = meshloc
        if self.dim == 2:
            degree = max(self.orders) - 1
        elif self.dim == 3:
            degree = reduce(lambda x, y: x + y, self.orders) - 3

        # Bspline control space
        self.V_control = fd.VectorFunctionSpace(self.mesh_r, "CG", degree)
        self.I_control = self.build_interpolation_matrix(self.V_control)

        # standard construction of ControlSpace
        self.mesh_r = mesh
        element = fd.VectorElement("CG", mesh.ufl_cell(), degree)
        self.V_r = fd.FunctionSpace(self.mesh_r, element)
        X = fd.SpatialCoordinate(self.mesh_r)
        self.id = fd.Function(self.V_r).interpolate(X)
        self.T = fd.Function(self.V_r, name="T")
        self.T.assign(self.id)
        self.mesh_m = fd.Mesh(self.T)
        self.V_m = fd.FunctionSpace(self.mesh_m, element)

        assert self.dim == self.mesh_r.geometric_dimension()

        # assemble correct interpolation matrix
        self.FullIFW = self.build_interpolation_matrix(self.V_r)

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
            # degree = order-1 # splev uses degree, not order
            assert level >= 1  # with level=1 only bdry Bsplines

            knots_01 = np.concatenate((np.zeros((order - 1,), dtype=float),
                                       np.linspace(0., 1., 2**level + 1),
                                       np.ones((order - 1,), dtype=float)))

            (xmin, xmax) = self.bbox[dim]
            knots = (xmax - xmin) * knots_01 + xmin
            self.knots.append(knots)
            # dimension of univariate spline spaces
            # the "-2" is because we want homogeneous Dir bc
            n = len(knots) - order - 2 * self.boundary_regularities[dim]
            assert n > 0
            self.n.append(n)

        # dimension of multivariate spline space
        N = reduce(lambda x, y: x * y, self.n)
        self.N = N

    def build_interpolation_matrix(self, V):
        """
        Construct the matrix self.FullIFW.

        The columns of self.FullIFW are the interpolant
        of (vectorial tensorized) Bsplines into V
        """
        # construct list of scalar univariate interpolation matrices
        interp_1d = self.construct_1d_interpolation_matrices(V)
        # construct scalar tensorial interpolation matrix
        self.IFWnnz = 0  # to compute sparsity pattern in parallel
        IFW = self.construct_kronecker_matrix(interp_1d)
        # interleave self.dim-many IFW matrices among each other
        self.FullIFWnnz = 0  # to compute sparsity pattern in parallel
        return self.construct_full_interpolation_matrix(IFW)

    def construct_1d_interpolation_matrices(self, V):
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
        x_fct = fd.SpatialCoordinate(self.mesh_r)  # used for x_int
        # compute self.M, x_int will be overwritten below
        x_int = fd.interpolate(x_fct[0], V.sub(0))
        self.M = x_int.vector().size()

        comm = self.comm

        u, v = fd.TrialFunction(V.sub(0)), fd.TestFunction(V.sub(0))
        mass_temp = fd.assemble(u * v * fd.dx)
        self.lg_map_fe = mass_temp.petscmat.getLGMap()[0]

        for dim in range(self.dim):

            order = self.orders[dim]
            knots = self.knots[dim]
            n = self.n[dim]

            # owned part of global problem
            local_n = n // comm.size + int(comm.rank < (n % comm.size))
            I = PETSc.Mat().create(comm=self.comm)
            I.setType(PETSc.Mat.Type.AIJ)
            lsize = x_int.vector().local_size()
            gsize = x_int.vector().size()
            I.setSizes(((lsize, gsize), (local_n, n)))

            I.setUp()
            x_int = fd.interpolate(x_fct[dim], V.sub(0))
            x = x_int.vector().get_local()
            for idx in range(n):
                coeffs = np.zeros(knots.shape, dtype=float)

                # impose boundary regularity
                coeffs[idx + self.boundary_regularities[dim]] = 1
                degree = order - 1  # splev uses degree, not order
                tck = (knots, coeffs, degree)

                values = splev(x, tck, der=0, ext=1)
                rows = np.where(values != 0)[0].astype(np.int32)
                values = values[rows]
                rows_is = PETSc.IS().createGeneral(rows)
                global_rows_is = self.lg_map_fe.applyIS(rows_is)
                rows = global_rows_is.array
                I.setValues(rows, [idx], values)

            I.assemble()  # lazy strategy for kron
            interp_1d.append(I)

        # from IPython import embed; embed()
        return interp_1d

    def vectorkron(self, v, w):
        """
        Compute the kronecker product of two sparse vectors.
        A sparse vector v satisfies: v[idx_] = data_; len_ = len(v)
        This code is an adaptation of scipy.sparse.kron()
        """
        idx1, data1, len1 = v
        idx2, data2, len2 = w

        # length of output vector
        lenout = len1*len2

        if len(data1) == 0 or len(data2) == 0:
            # if a vector is zero, the output is the zero vector
            idxout = []
            dataout = []
            return (idxout, dataout, lenout)
        else:
            # rewrite as column vector, multiply, and add row vector
            idxout = (idx1.reshape(len(data1), 1) * len2) + idx2
            dataout = data1.reshape(len(data1), 1) * data2
            return (idxout.reshape(-1), dataout.reshape(-1), lenout)

    def construct_kronecker_matrix(self, interp_1d):
        """
        Construct the tensorized interpolation matrix.

        Do this by computing the kron product of the rows of
        the 1d univariate interpolation matrices.
        In the future, this may be done matrix-free.
        """
        # this is one of the two bottlenecks that slow down initiating Bsplines
        IFW = PETSc.Mat().create(self.comm)
        IFW.setType(PETSc.Mat.Type.AIJ)

        comm = self.comm
        # owned part of global problem
        local_N = self.N // comm.size + int(comm.rank < (self.N % comm.size))
        (lsize, gsize) = interp_1d[0].getSizes()[0]
        IFW.setSizes(((lsize, gsize), (local_N, self.N)))

        # guess sparsity pattern from interp_1d[0]
        for row in range(lsize):
            row = self.lg_map_fe.apply([row])[0]
            nnz_ = len(interp_1d[0].getRow(row)[0])  # length of nnz-array
            self.IFWnnz = max(self.IFWnnz, nnz_**self.dim)
        IFW.setPreallocationNNZ(self.IFWnnz)
        IFW.setUp()

        for row in range(lsize):
            row = self.lg_map_fe.apply([row])[0]
            M = [[A.getRow(row)[0],
                  A.getRow(row)[1], A.getSize()[1]] for A in interp_1d]
            M = reduce(self.vectorkron, M)
            columns, values, length = M
            IFW.setValues([row], columns, values)

        IFW.assemble()
        return IFW

    def construct_full_interpolation_matrix(self, IFW):
        """
        Assemble interpolation matrix for vectorial tensorized spline space.
        """
        # this is one of the two bottlenecks that slow down initiating Bsplines
        FullIFW = PETSc.Mat().create(self.comm)
        FullIFW.setType(PETSc.Mat.Type.AIJ)

        # set proper matrix sizes
        d = self.dim
        free_dims = list(set(range(self.dim)) - set(self.fixed_dims))
        dfree = len(free_dims)
        ((lsize, gsize), (lsize_spline, gsize_spline)) = IFW.getSizes()
        FullIFW.setSizes(((d * lsize, d * gsize),
                          (dfree * lsize_spline, dfree * gsize_spline)))

        # (over)estimate sparsity pattern using row with most nonzeros
        # possible memory improvement: allocate precise sparsity pattern
        # row by row (but this needs nnzdiagonal and nnzoffidagonal;
        # not straightforward to do)
        global_rows = self.lg_map_fe.apply([range(lsize)])
        for ii, row in enumerate(range(lsize)):
            row = global_rows[row]
            self.FullIFWnnz = max(self.FullIFWnnz, len(IFW.getRow(row)[1]))
        FullIFW.setPreallocationNNZ(self.FullIFWnnz)

        # preallocate matrix
        FullIFW.setUp()

        # fill matrix by blowing up entries from IFW to do the right thing
        # on vector fields (it's not just a block matrix: values are
        # interleaved as this is how firedrake handles vector fields)
        innerloop_idx = [[i, free_dims[i]] for i in range(dfree)]
        for row in range(lsize):  # for every FE dof
            row = self.lg_map_fe.apply([row])[0]
            # extract value of all tensorize Bsplines at this dof
            (cols, vals) = IFW.getRow(row)
            expandedcols = dfree * cols
            for j, dim in innerloop_idx:
                FullIFW.setValues([d * row + dim],   # global row
                                  expandedcols + j,  # global column
                                  vals)

        FullIFW.assemble()
        return FullIFW

    def restrict(self, residual, out):
        with residual.dat.vec as w:
            self.FullIFW.multTranspose(w, out.vec_wo())

    def interpolate(self, vector, out):
        with out.dat.vec as w:
            self.FullIFW.mult(vector.vec_ro(), w)

    def get_zero_vec(self):
        vec = self.FullIFW.createVecRight()
        return vec

    def get_space_for_inner(self):
        return (self.V_control, self.I_control)

    def visualize_control(self, q, out):
        with out.dat.vec_wo as outp:
            self.I_control.mult(q.vec_wo(), outp)

    def store(self, vec, filename="control.dat"):
        """
        Store the vector to a file to be reused in a later computation.
        DumbCheckpoint requires that the mesh, FunctionSpace and parallel
        decomposition are identical between store and load.

        """
        viewer = PETSc.Viewer().createBinary(filename, mode="w")
        viewer.view(vec.vec_ro())

    def load(self, vec, filename="control.dat"):
        """
        Load a vector from a file.
        DumbCheckpoint requires that the mesh, FunctionSpace and parallel
        decomposition are identical between store and load.
        """
        viewer = PETSc.Viewer().createBinary(filename, mode="r")
        vec.vec_wo().load(viewer)


class WaveletControlSpace(BsplineControlSpace):
    """
    ControlSpace based on cartesian tensorized biorthogonal spline-wavelets.
    """

    def __init__(self, mesh, bbox, orders, dual_orders, levels, fixed_dims=[],
                 homogeneous_bc=None, tol=None):
        """
        bbox: a list of tuples describing [(xmin, xmax), (ymin, ymax), ...]
              of a Cartesian grid that extends around the shape to be
              optimised
        orders: describe the orders (one integer per geometric dimension)
                of the tensor-product spline-wavelet basis. A univariate
                wavelet has order "o" if it is a piecewise polynomial of
                degree "o-1".
        dual_orders: describe the orders (one integer per geometric dimension)
                     of the dual basis. For each dimension, the order "o" and
                     the dual order "o_t" must satisfy that
                     (o + o_t) mod 2 == 0 and o_t >= o. It is set to the same
                     value as orders by default.
        levels: describe the subdivision levels (one integers per
                geometric dimension) used to construct the knots of
                univariate B-splines
        fixed_dims: dimensions in which the deformation should be zero
        homogeneous_bc: impose homogeneous Dirichlet boundary conditions on
                        wavelets for each dimension. True for all boundaries
                        by default.
        tol: threshold for selecting basis functions
        """
        if homogeneous_bc is None:
            # set homogeneous Dirichlet bc on all boundaries
            homogeneous_bc = [True] * len(bbox)

        if dual_orders is None:
            # use same orders for both primal and dual wavelets
            self.dual_orders = orders
        else:
            self.dual_orders = dual_orders

        # construct transformation matrices for univariate wavelets
        # based on DOI:10.1007/s00025-009-0008-6
        self.j0 = []
        self.mat = []
        self.n_split = []
        for dim in range(len(bbox)):
            d = orders[dim]
            d_t = dual_orders[dim]
            J = levels[dim]
            bc = homogeneous_bc[dim]

            a = self.primal_refinement_coeffs(d)
            a_t = self.dual_refinement_coeffs(d, d_t)
            ML = self.primal_ML(d, bc)
            ML_t = self.dual_ML(d, d_t, a, a_t, ML, bc)
            GL = self.primal_GL(d, d_t, a, a_t, ML, ML_t, bc)
            T = self.transformation_matrix(J, d, d_t, a, a_t, ML, GL, bc)
            self.mat.append(T)

        super().__init__(mesh, bbox, orders, levels, fixed_dims,
                         homogeneous_bc)

        self.free_dims = list(set(range(self.dim)) - set(self.fixed_dims))
        self.dfree = len(self.free_dims)
        self.tol = tol

    def primal_refinement_coeffs(self, d):
        """
        Compute the refinement coefficients for the primal scaling function
        on real line.
        """
        l1 = -floor(d / 2)
        l2 = ceil(d / 2)
        a = 2**(1-d) * np.array([binom(d, k - l1) for k in range(l1, l2 + 1)])
        return a

    def primal_ML(self, d, bc):
        """
        Compute the block of refinement matrix for primal left boundary
        scaling functions.
        """
        knots = np.concatenate((np.zeros(d - 1), np.arange(3 * d - 3)))
        x = np.arange(2 * d - 2)

        B1 = np.empty((2 * d - 2, d - 1))
        for k in range(d - 1):
            coeffs = np.zeros(3 * d - 4)
            coeffs[k] = 1.
            tck = (knots, coeffs, d - 1)
            B1[:, k] = splev(x / 2, tck, der=0, ext=1)

        B2 = np.empty((2 * d - 2, 2 * d - 2))
        for k in range(2 * d - 2):
            coeffs = np.zeros(3 * d - 4)
            coeffs[k] = 1.
            tck = (knots, coeffs, d - 1)
            B2[:, k] = splev(x, tck, der=0, ext=1)

        ML = np.linalg.solve(B2, B1)
        ML[np.abs(ML) < 1e-9] = 0.
        if bc:
            ML = ML[1:, 1:]
        return ML

    def primal_A(self, j, d, a):
        """
        Compute the block of refinement matrix for primal inner scaling
        functions at level j.
        """
        n = a.size
        A = np.zeros((2**(j+1) - d + 1, 2**j - d + 1))
        for k in range(A.shape[1]):
            A[2*k:2*k+n, k] = a
        return A

    def primal_M0(self, j, d, a, ML, bc):
        """
        Construct the refinement matrix for primal scaling functions
        at level j.
        """
        A = self.primal_A(j, d, a)

        M0 = np.zeros((2**(j+1) + d - 1 - 2 * bc, 2**j + d - 1 - 2 * bc))
        m, n = M0.shape
        mL, nL = ML.shape
        M0[:mL, :nL] = ML
        M0[nL:m-nL, nL:n-nL] = A
        M0[m-mL:, n-nL:] = ML[::-1, ::-1]

        return 1 / np.sqrt(2) * M0

    def dual_refinement_coeffs(self, d, d_t):
        """
        Compute the refinement coefficients for the dual scaling function
        on real line.
        """
        l1_t = -floor(d / 2) - d_t + 1
        l2_t = ceil(d / 2) + d_t - 1
        K = (d + d_t) // 2

        def entry(k):
            res = 0
            for n in range(K):
                for i in range(2 * n + 1):
                    res += 2**(1 - d_t - 2 * n) * (-1)**(n + i) \
                           * binom(d_t, k + floor(d_t / 2) - i + n) \
                           * binom(K - 1 + n, n) * binom(2 * n, i)
            return res
        a_t = np.array([entry(k) for k in range(l1_t, l2_t + 1)])
        return a_t

    def dual_ML(self, d, d_t, a, a_t, ML, bc):
        """
        Compute the block of refinement matrix for dual left boundary
        scaling functions.
        """
        l1 = -floor((a.size - 1) / 2)
        l2 = ceil((a.size - 1) / 2)
        l1_t = -floor((a_t.size - 1) / 2)
        l2_t = ceil((a_t.size - 1) / 2)
        if bc and d == 2:
            ML_t = np.zeros((5 * d_t,  2 * d_t))
        else:
            ML_t = np.zeros((2 * d + 3 * d_t - 5 - bc, d + d_t - 2 - bc))
        mL, nL = ML_t.shape

        m, n = ML.shape
        ML_full = np.zeros_like(ML_t)
        ML_full[:m, :n] = ML
        for k in range(n, nL):
            ML_full[2*k-n:2*k-n+d+1, k] = a
        ML = ML_full

        # Compute block of ML_t corresponding to k = d-2, ..., d+2*d_t-3

        # Compute alpha_{0,r}
        alpha0 = np.zeros(d_t)
        alpha0[0] = 1
        for r in range(1, d_t):
            for k in range(l1, l2 + 1):
                sum = 0
                for s in range(r):
                    sum += binom(r, s) * k**(r-s) * alpha0[s]
                alpha0[r] += a[k-l1] * sum
            alpha0[r] /= (2**(r+1) - 2)

        # Compute alpha_{k,r}
        def alpha(k, r):
            res = 0
            for i in range(r + 1):
                res += binom(r, i) * k**i * alpha0[r-i]
            return res

        def compute_gramian():
            n = ML.shape[1]
            UL = ML[:n, :]
            LL = ML[n:, :]
            UL_t = ML_t[:n, :]
            LL_t = ML_t[n:, :]
            lhs = 2 * np.identity(n**2) - np.kron(UL_t.T, UL.T)
            rhs = (LL.T @ LL_t).reshape(-1, order='F')
            gamma = np.linalg.solve(lhs, rhs)
            return gamma.reshape((n, n), order='F')

        if bc and d == 2:
            # Compute beta_{n,r}
            def beta(n, r):
                res = 0
                for k in range(ceil((n-d_t) / 2), d_t + 1):
                    res += alpha(k, r) * a_t[n-2*k+d_t]
                return res

            ML_t[:d_t, :d_t] = np.diag([2**(-r) for r in range(d_t)])
            ML_t[d_t, :d_t] = np.array(
                [2**(-r) * alpha(d_t + 1, r) for r in range(d_t)])
            ML_t[d_t+1:3*d_t, :d_t] = np.array(
                [[beta(n+d_t+2, r) for r in range(d_t)]
                 for n in range(2 * d_t - 1)])
            for k in range(d_t, 2 * d_t):
                ML_t[2*k-d_t+1:2*k+d_t+2, k] = a_t

            # Biorthogonalize

            gramian = compute_gramian()
            ML_t[:2*d_t, :] = gramian @ ML_t[:2*d_t, :]
            ML_t = np.linalg.solve(gramian.T, ML_t.T).T
            ML_t = ML_t[:3*d_t, :d_t]

        else:
            # Compute beta_{n,r}
            def beta(n, r):
                res = 0
                for k in range(ceil((n-l2_t) / 2), -l1_t):
                    res += alpha(k, r) * a_t[n-2*k-l1_t]
                return res

            def divided_diff(f, t):
                if t.size == 1:
                    return f(t[0])
                return (divided_diff(f, t[1:]) - divided_diff(f, t[:-1])) \
                    / (t[-1] - t[0])

            D1 = np.zeros((d_t, d_t))
            D2 = np.zeros((d_t, d_t))
            D3 = np.zeros((d_t, d_t))
            k0 = -l1_t - 1
            for n in range(d_t):
                for k in range(n + 1):
                    D1[n, k] = binom(n, k) * alpha0[n-k]
                    D2[n, k] = binom(n, k) * k0**(n-k) * (-1)**k
                    D3[n, k] = factorial(k) \
                        * divided_diff(lambda x: x**n, np.arange(k + 1))
            D_t = D1 @ D2 @ D3
            rhs = np.empty((d_t, d + 3 * d_t - 3))
            rhs[:, :d_t] = \
                np.diag([2**(-r) for r in range(d_t)]) @ D_t[:, ::-1]
            rhs[:, d_t:] = np.array(
                [[beta(n-l1_t, r) for n in range(d + 2 * d_t - 3)]
                 for r in range(d_t)])
            ML_t[d-2-bc:, d-2-bc:] = np.linalg.solve(D_t, rhs)[::-1, :].T

            # Compute block of ML_t corresponding to k = 0, ..., d-3

            gramian_full = np.identity(mL)
            for k in range(d - 3 - bc, -1, -1):
                gramian_full[:nL, :nL] = compute_gramian()
                B_k = ML[:, :k+d].T @ gramian_full[:, k+1:2*k+d+1] / 2.

                delta = np.zeros(k + d)
                delta[k] = 1
                ML_t[k+1:2*k+d+1, k] = np.linalg.solve(B_k, delta)

            # Biorthogonalize

            gramian = np.triu(compute_gramian())
            gramian[:d-2-bc, :d-2-bc] = np.identity(d - 2 - bc)
            ML_t[:d+d_t-2-bc, :] = gramian @ ML_t[:d+d_t-2-bc, :]
            ML_t = np.linalg.solve(gramian.T, ML_t.T).T

        ML_t[np.abs(ML_t) < 1e-9] = 0.
        return ML_t

    def dual_A(self, j, d, d_t, a_t, bc):
        """
        Compute the block of refinement matrix for dual inner scaling
        functions at level j.
        """
        n = a_t.size
        if bc and d == 2:
            A_t = np.zeros((2**(j+1) - 2 * d_t - 3,
                            2**j - 2 * d_t - 1))
        else:
            A_t = np.zeros((2**(j+1) - d - 2 * d_t + 3,
                            2**j - d - 2 * d_t + 3))
        for k in range(A_t.shape[1]):
            A_t[2*k:2*k+n, k] = a_t
        return A_t

    def dual_M0(self, j, d, d_t, a_t, ML_t, bc):
        """
        Construct the refinement matrix for dual scaling functions
        at level j.
        """
        A_t = self.dual_A(j, d, d_t, a_t, bc)

        M0_t = np.zeros((2**(j+1) + d - 1 - 2 * bc, 2**j + d - 1 - 2 * bc))
        m, n = M0_t.shape
        mL, nL = ML_t.shape
        shift = (m - A_t.shape[0]) // 2
        M0_t[:mL, :nL] = ML_t
        M0_t[shift:m-shift, nL:n-nL] = A_t
        M0_t[m-mL:, n-nL:] = ML_t[::-1, ::-1]

        return 1 / np.sqrt(2) * M0_t

    def primal_GL(self, d, d_t, a, a_t, ML, ML_t, bc):
        """
        Compute the block of refinement matrix for primal left boundary
        wavelets.
        """
        if bc and d == 2:
            j0 = ceil(np.log2(3 / 2 * d_t + 1) + 1)
        else:
            j0 = ceil(np.log2(d + d_t - 2) + 1)

        # initial completion
        l1 = -floor((a.size - 1) / 2)
        l2 = ceil((a.size - 1) / 2)
        p = 2**j0 - d + 1
        q = 2**(j0+1) - d + 1

        mL, nL = ML.shape
        P = np.identity(q + 2 * nL)
        P[:mL, :nL] = ML
        P[q+2*nL-mL:, q+nL:] = ML[::-1, ::-1]

        ML_inv = np.linalg.inv(P[:mL, :mL])
        P_inv = np.identity(q + 2 * nL)
        P_inv[:mL, :mL] = ML_inv
        P_inv[q+2*nL-mL:, q+2*nL-mL:] = ML_inv[::-1, ::-1]

        A = self.primal_A(j0, d, a)
        H_inv = np.identity(q)
        for i in range(d):
            if i % 2 == 0:
                m = i // 2
                v = A[m, 0] / A[m+1, 0]
                H_i = np.identity(q)
                H_i_inv = np.identity(q)
                rows = np.arange(m % 2, q, 2)
                cols = np.arange(m % 2 + 1, q, 2)
                rows = rows[:cols.size]
                H_i[rows, cols] = -v
                H_i_inv[rows, cols] = v
            else:
                m = (i - 1) // 2
                v = A[d-m, 0] / A[d-m-1, 0]
                H_i = np.identity(q)
                H_i_inv = np.identity(q)
                rows = np.arange(q - m % 2, 0, -2) - 1
                cols = np.arange(q - m % 2 - 1, 0, -2) - 1
                rows = rows[:cols.size]
                H_i[rows, cols] = -v
                H_i_inv[rows, cols] = v
            A = H_i @ A
            H_inv = H_inv @ H_i_inv

        b = A[l2, 0]
        rows = np.arange(p) * 2 + l2 - 1
        cols = np.arange(p)
        F = np.zeros_like(A)
        F[rows, cols] = 1

        F_hat = np.zeros((q + 2 * nL, 2**j0))
        F_hat[nL:l2-1+nL, :l2-1] = np.identity(l2 - 1)
        F_hat[nL:q+nL, l2-1:p+l2-1] = F
        F_hat[q+nL+l1:q+nL, 2**j0+l1:] = np.identity(-l1)

        H_hat_inv = np.identity(q + 2 * nL)
        H_hat_inv[nL:q+nL, nL:q+nL] = H_inv

        M1 = (-1)**(d+1) * np.sqrt(2) / b * P @ H_hat_inv @ F_hat

        # stable completion
        M0 = self.primal_M0(j0, d, a, ML, bc)
        M0_t = self.dual_M0(j0, d, d_t, a_t, ML_t, bc)
        M1 = M1 - M0 @ M0_t.T @ M1

        if bc and d == 2:
            GL = np.sqrt(2) * M1[:2*d_t+1, :d_t//2]
        else:
            GL = np.sqrt(2) * M1[:2*(d+d_t-2)-bc, :(d+d_t-2)//2]

        GL[np.abs(GL) < 1e-9] = 0.
        return GL

    def primal_B(self, j, d, d_t, a_t):
        """
        Compute the block of refinement matrix for primal inner wavelets
        at level j.
        """
        l1_t = -floor((a_t.size - 1) / 2)
        l2_t = ceil((a_t.size - 1) / 2)
        sgn = (-1)**np.abs(np.arange(1 - l2_t, 2 - l1_t))
        b = sgn * np.flip(a_t)

        B = np.zeros((2**(j+1) - d + 1, 2**j - d - d_t + 2))
        for k in range(2**(j-1) - (d + d_t - 2) // 2):
            B[2*k:2*k+d+2*d_t-1, k] = b
        B += B[::-1, ::-1]
        return B

    def primal_M1(self, j, d, d_t, a_t, GL, bc):
        """
        Construct the refinement matrix for primal wavelets at level j.
        """
        B = self.primal_B(j, d, d_t, a_t)

        M1 = np.zeros((2**(j+1) + d - 1 - 2 * bc, 2**j))
        m, n = M1.shape
        mL, nL = GL.shape
        shift = (m - B.shape[0]) // 2
        M1[:mL, :nL] = GL
        M1[shift:m-shift, nL:n-nL] = B
        M1[m-mL:, n-nL:] = GL[::-1, ::-1]

        return 1 / np.sqrt(2) * M1

    def transformation_matrix(self, J, d, d_t, a, a_t, ML, GL, bc):
        """
        Construct the transformation matrix that transforms univariate
        B-splines at level J to a multi-scale wavelet basis.
        """
        if bc and d == 2:
            j0 = ceil(np.log2(3 / 2 * d_t + 1) + 1)
        else:
            j0 = ceil(np.log2(d + d_t - 2) + 1)
        self.j0.append(j0)

        T = np.identity(2**J + d - 1 - 2 * bc)
        for j in range(j0, J):
            M0 = self.primal_M0(j, d, a, ML, bc)
            M1 = self.primal_M1(j, d, d_t, a_t, GL, bc)
            M = np.hstack((M0, M1))
            n = M.shape[0]
            T[:n, :n] = M @ T[:n, :n]

        n = 2**j0 + d - 1 - 2 * bc
        T_split = [T[:, :n]]
        n_split = [n]
        offset = n
        for j in range(j0, J):
            n = 2**j
            T_split.append(T[:, offset:offset+n])
            n_split.append(n)
            offset += n
        self.n_split.append(n_split)
        return T_split

    def construct_1d_interpolation_matrices(self, V):
        interp_1d = []

        # this code is correct but can be made more beautiful
        # by replacing x_fct with self.id
        x_fct = fd.SpatialCoordinate(self.mesh_r)  # used for x_int
        # compute self.M, x_int will be overwritten below
        x_int = fd.interpolate(x_fct[0], V.sub(0))
        self.M = x_int.vector().size()

        comm = self.comm

        u, v = fd.TrialFunction(V.sub(0)), fd.TestFunction(V.sub(0))
        mass_temp = fd.assemble(u * v * fd.dx)
        self.lg_map_fe = mass_temp.petscmat.getLGMap()[0]

        for dim in range(self.dim):
            order = self.orders[dim]
            level = self.levels[dim]
            T = self.mat[dim]
            bc = self.boundary_regularities[dim]
            knots = self.knots[dim]
            m = 2**level + order - 1

            interp_sub = []
            for T_sub in T:
                n = T_sub.shape[1]

                # owned part of global problem
                local_n = n // comm.size + int(comm.rank < (n % comm.size))
                I = PETSc.Mat().create(comm=self.comm)
                I.setType(PETSc.Mat.Type.AIJ)
                lsize = x_int.vector().local_size()
                gsize = x_int.vector().size()
                I.setSizes(((lsize, gsize), (local_n, n)))

                I.setUp()
                x_int = fd.interpolate(x_fct[dim], V.sub(0))
                x = x_int.vector().get_local()
                for idx in range(n):
                    coeffs = np.zeros(m, dtype=float)

                    # impose boundary regularity
                    coeffs[bc:m-bc] = T_sub[:, idx]
                    degree = order - 1  # splev uses degree, not order
                    tck = (knots, coeffs, degree)

                    values = splev(x, tck, der=0, ext=1)
                    rows = np.where(values != 0)[0].astype(np.int32)
                    values = values[rows]
                    rows_is = PETSc.IS().createGeneral(rows)
                    global_rows_is = self.lg_map_fe.applyIS(rows_is)
                    rows = global_rows_is.array
                    I.setValues(rows, [idx], values)

                I.assemble()  # lazy strategy for kron
                interp_sub.append(I)
            interp_1d.append(interp_sub)

        # from IPython import embed; embed()
        return interp_1d

    def construct_kronecker_matrix(self, interp_1d):
        """
        Construct the block tensorized interpolation matrix.

        The implementation is based on super().construct_kronecker_matrix
        but computes the kron product of the rows of the 1d univariate
        interpolation submatrices. The resulting blocks are interpolation
        matrices for tensorized wavelets on different levels.
        """
        IFW = PETSc.Mat().create(self.comm)
        IFW.setType(PETSc.Mat.Type.AIJ)

        comm = self.comm
        # owned part of global problem
        local_N = self.N // comm.size + int(comm.rank < (self.N % comm.size))
        (lsize, gsize) = interp_1d[0][0].getSizes()[0]
        IFW.setSizes(((lsize, gsize), (local_N, self.N)))

        # guess sparsity pattern
        for row in range(lsize):
            row = self.lg_map_fe.apply([row])[0]
            nnz_ = 1
            for interp_sub in interp_1d:
                nnz_ *= len(interp_sub[-1].getRow(row)[0]) * len(interp_sub)
            self.IFWnnz = max(self.IFWnnz, nnz_)
        IFW.setPreallocationNNZ(self.IFWnnz)
        IFW.setUp()

        interp_1d = list(product(*interp_1d))
        for row in range(lsize):
            offset = 0
            row = self.lg_map_fe.apply([row])[0]
            for blocks in interp_1d:
                M = [[A.getRow(row)[0],
                      A.getRow(row)[1], A.getSize()[1]] for A in blocks]
                M = reduce(self.vectorkron, M)
                columns, values, length = M
                if len(columns):
                    columns += offset
                IFW.setValues([row], columns, values)
                offset += length

        IFW.assemble()
        return IFW

    def assign_inner_product(self, inner_product):
        """Normalize basis functions."""
        A = inner_product.A
        D = A.getDiagonal()
        D.sqrtabs()
        D = 1 / D
        self.I_control.diagonalScale(R=D)
        self.FullIFW.diagonalScale(R=D)
        A.diagonalScale(L=D, R=D)

    def visualize_control(self, q, filename="control"):
        """
        Visualize distribution of wavelet coefficients.

        For each dimension of the deformation, the tensorized wavelet basis is
        a direct sum of subspaces with wavelet bases on different levels, and
        a subfigure is plotted per subspace. At every point of the domain,
        the contributions from the wavelets whose supports cover that point
        are counted.
        """
        if self.dim != 2:
            raise ValueError("Only 2D visualization is supported.")

        supp_1d = []
        nx_1d = []
        for dim in range(self.dim):
            d = self.orders[dim]
            d_t = self.dual_orders[dim]
            j0 = self.j0[dim]
            J = self.levels[dim]
            bc = self.boundary_regularities[dim]
            n = self.n_split[dim]
            m = (d + d_t - 2) // 2  # number of boundary wavelets
            supp_split = []
            nx = []

            # compute supports of scaling functions
            supp = []
            for k in range(n[0]):
                supp.append((max(k+bc-d+1, 0),
                             min(k+bc+1, 2**j0)))
            supp_split.append(supp)
            nx.append(2**j0)

            # compute supports of wavelets
            for j in range(j0, J):
                supp = []
                if bc and d == 2:
                    for k in range(d_t // 2):
                        supp.append((0, d_t + 1))
                    for k in range(d_t // 2, 2**j - d_t // 2):
                        supp.append((max(k-m, 0),
                                     min(k-m+d+d_t-1, 2**j)))
                    for k in range(2**j - d_t // 2, 2**j):
                        supp.append((2**j - (d_t + 1), 2**j))
                else:
                    for k in range(m):
                        supp.append((0, d + d_t - 2))
                    for k in range(m, 2**j - m):
                        supp.append((max(k-m, 0),
                                     min(k-m+d+d_t-1, 2**j)))
                    for k in range(2**j - m, 2**j):
                        supp.append((2**j - (d + d_t - 2), 2**j))
                supp_split.append(supp)
                nx.append(2**j)
            supp_1d.append(supp_split)
            nx_1d.append(nx)

        import matplotlib.pyplot as plt
        import matplotlib.colors as colors
        fig_rows, fig_cols = len(supp_1d[1]), len(supp_1d[0])
        figs = []
        axs = []
        ims = []
        for _ in self.free_dims:
            fig, ax = plt.subplots(fig_rows, fig_cols)
            figs.append(fig)
            axs.append(ax)
            ims.append(None)

        v_sq = q.vec_ro().array**2
        extent = [self.bbox[0][0], self.bbox[0][1],
                  self.bbox[1][0], self.bbox[1][1]]
        k = 0
        for j in range(fig_cols):
            supp_x = supp_1d[0][j]
            for i in range(fig_rows):
                supp_y = supp_1d[1][i]
                data = [np.zeros((nx_1d[1][i], nx_1d[0][j]))
                        for _ in self.free_dims]

                # add squared magnitudes of coefficients
                # to supports of corresponding wavelets
                for xmin, xmax in supp_x:
                    for ymin, ymax in supp_y:
                        for d in range(self.dfree):
                            data[d][ymin:ymax, xmin:xmax] += v_sq[k]
                            k += 1

                for d in range(self.dfree):
                    ax = axs[d][i, j]
                    data_ = np.sqrt(data[d][::-1, :])
                    # avoid undefined values on a log scale
                    data_[data_ < 1e-12] = 1e-12
                    ims[d] = ax.imshow(data_, extent=extent,
                                       norm=colors.LogNorm(vmin=1e-3, vmax=1))
                    if i == 0:
                        j0 = self.j0[0]
                        ax.xaxis.set_label_position('top')
                        if j == 0:
                            ax.set_xlabel(rf"$V_{j0}$")
                        else:
                            ax.set_xlabel(rf"$W_{j0+j-1}$")
                    if j == 0:
                        j0 = self.j0[1]
                        if i == 0:
                            ax.set_ylabel(rf"$V_{j0}$")
                        else:
                            ax.set_ylabel(rf"$W_{j0+i-1}$")

        for d in range(self.dfree):
            fig = figs[d]
            ax = axs[d]
            im = ims[d]
            plt.setp(ax, xticks=[], yticks=[])
            fig.colorbar(im, ax=ax)
            fig.savefig(filename + str(self.free_dims[d]) + ".png", dpi=150,
                        bbox_inches="tight")

    def thresholding(self, q):
        """Remove wavelet coefficients with small magnitudes."""
        v_sq = q.vec_ro().array**2

        mask = np.full_like(v_sq, False, dtype=bool)
        c = v_sq.sum() * (1 - self.tol**2)
        ind_sorted = np.argsort(-v_sq)
        sum = 0.
        for i in range(v_sq.size):
            ii = ind_sorted[i]
            sum += v_sq[ii]
            mask[ii] = True
            if sum > c:
                break
        n_sf = reduce(lambda x, y: x * y,
                      [n_split[0] for n_split in self.n_split]) * self.dfree
        mask[:n_sf] = True  # keep all scaling functions
        indices = np.where(~mask)[0].astype(np.int32)
        q.vec_wo().setValues(indices, np.zeros_like(indices))
        q.vec_wo().assemble()
        # print("Filtered {:.2%} entries".format(indices.size / v_sq.size))


class ControlVector(ROL.Vector):
    """
    A ControlVector is a variable in the ControlSpace.

    The data of a control vector is a PETSc.vec stored in self.vec.
    If this data corresponds also to a fd.Function, the firedrake wrapper
    around self.vec is stored in self.fun (otherwise, self.fun = None).

    A ControlVector is a ROL.Vector and thus needs the following methods:
    plus, scale, clone, dot, axpy, set.
    """

    def __init__(self, controlspace: ControlSpace, inner_product: InnerProduct,
                 data=None, boundary_extension=None):
        super().__init__()
        self.controlspace = controlspace
        self.inner_product = inner_product
        self.boundary_extension = boundary_extension

        if data is None:
            data = controlspace.get_zero_vec()

        self.data = data
        if isinstance(data, fd.Function):
            self.fun = data
        else:
            self.fun = None

    def from_first_derivative(self, fe_deriv):
        if self.boundary_extension is not None:
            residual_smoothed = fe_deriv.copy(deepcopy=True)
            p1 = fe_deriv
            p1 *= -1
            self.boundary_extension.solve_homogeneous_adjoint(
                p1, residual_smoothed)
            self.boundary_extension.apply_adjoint_action(
                residual_smoothed, residual_smoothed)
            residual_smoothed -= p1
            self.controlspace.restrict(residual_smoothed, self)
        else:
            self.controlspace.restrict(fe_deriv, self)

    def to_coordinatefield(self, out):
        self.controlspace.interpolate(self, out)
        if self.boundary_extension is not None:
            self.boundary_extension.extend(out, out)

    def apply_riesz_map(self):
        """
        Maps this vector into the dual space.
        Overwrites the content.
        """
        self.inner_product.riesz_map(self, self)
        if isinstance(self.controlspace, WaveletControlSpace) and \
                self.controlspace.tol:
            self.controlspace.thresholding(self)

    def vec_ro(self):
        if isinstance(self.data, fd.Function):
            with self.data.dat.vec_ro as v:
                return v
        else:
            return self.data

    def vec_wo(self):
        if isinstance(self.data, fd.Function):
            with self.data.dat.vec_wo as v:
                return v
        else:
            return self.data

    def plus(self, v):
        vec = self.vec_wo()
        vec += v.vec_ro()

    def scale(self, alpha):
        vec = self.vec_wo()
        vec *= alpha

    def clone(self):
        """
        Returns a zero vector of the same size of self.

        The name of this method is misleading, but it is dictated by ROL.
        """
        res = ControlVector(self.controlspace, self.inner_product,
                            boundary_extension=self.boundary_extension)
        # res.set(self)
        return res

    def dot(self, v):
        """Inner product between self and v."""
        return self.inner_product.eval(self, v)

    def norm(self):
        return self.dot(self)**0.5

    def axpy(self, alpha, x):
        vec = self.vec_wo()
        vec.axpy(alpha, x.vec_ro())

    def set(self, v):
        vec = self.vec_wo()
        v.vec_ro().copy(vec)

    def __str__(self):
        """String representative, so we can call print(vec)."""
        return self.vec_ro()[:].__str__()
