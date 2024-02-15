from .innerproduct import InnerProduct
import ROL
import firedrake as fd
import firedrake.adjoint as fda
from icecream import ic
ic.configureOutput(includeContext=True) 

__all__ = ["FeControlSpace", "FeMultiGridControlSpace",
           "BsplineControlSpace", "ControlVector", "CmControlSpace"]

# new imports for splines
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
        self.V_r is a Firedrake vectorial Lagrangian finite element
            space on mesh_r
        self.id is the element of V_r that satisfies id(x) = x for every x
        self.T is the interpolant of a ControlSpace variable in self.V_r
        self.mesh_m is the mesh that corresponds to self.T (moved domain)
        self.V_m is the Firedrake vectorial Lagrangian finite element
            space on mesh_m
        self.V_m_dual is the dual of self.V_m
        self.inner_product is the inner product of the ControlSpace

    Key idea: solve state and adjoint equations on mesh_m. Then, evaluate
    shape derivatives along directions in V_m, transplant this to V_r,
    restrict to ControlSpace, compute the update (using inner_product),
    and interpolate it into V_r to update mesh_m.

    Note: transplant to V_r means creating a cofunction in V_r_dual and using
    the values of the directional shape derivative as coefficients. Since
    Lagrangian finite elements are parametric, no transformation matrix is
    required by this operation.
    """

    def restrict(self, residual, out):
        """
        Restrict from self.V_r_dual into ControlSpace

        Input:
        residual: fd.Cofunction, is a variable in self.V_r_dual
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
        # more than once, in that case we don't want to solve the PDE again.
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


class CmControlSpace(ControlSpace):
    def __init__(self, mesh_r, mesh_m, indicator):
        # Create mesh_r and V_r
        self.mesh_r = mesh_r
        self.I = indicator

        element = self.mesh_r.coordinates.function_space().ufl_element()
        family = element.family()
        degree = element.degree()
        self.V_r = fd.VectorFunctionSpace(self.mesh_r, "CG", 1)
        self.V_r_dual = self.V_r.dual()

        # Create self.id and self.T, self.mesh_m, and self.V_m.
        self.X = fd.SpatialCoordinate(self.mesh_r)
        self.id = fd.interpolate(self.X, self.V_r)
        self.T = fd.Function(self.V_r, name="T")
        self.T.assign(self.id)
        self.mesh_m = mesh_m
        self.V_m = fd.VectorFunctionSpace(self.mesh_m, "CG", 1)
        self.V_m_dual = self.V_m.dual()

        # Scalar function space to hold the control vector p0
        self.P = fd.FunctionSpace(self.mesh_r, "CG", 1)
        self.P_dual = self.P.dual()
        
        v = fd.TestFunction(self.V_r)
        u = fd.TrialFunction(self.V_r)

        self.dphi = fd.Function(self.V_r, name="dphi")
        phi = self.X + self.dphi

        n = fd.FacetNormal(self.mesh_r)
        normal = self.I("+")*n("+") + self.I("-")*n("-")

        self.p0 = fd.Function(self.P)
        self.J = fd.grad(phi)
        self.Jit = fd.inv(self.J.T)
        self.a = (fd.inner(u, v) + fd.inner(self.Jit * fd.grad(v), self.Jit * fd.grad(u))) \
            * fd.det(self.J) * fd.dx

        # TODO: move ds(11) into variable?
        self.L = fd.inner(self.Jit("+") * normal, self.p0("+") * v("+")) * fd.dS(11)
        
        # TODO: move bcs into constructor
        self.bcs = [fd.DirichletBC(self.V_r, fd.Constant((0, 0)), "on_boundary")]
        self.u0 = fd.Function(self.V_r, name="u0")
        self.residual = fd.Cofunction(self.V_r_dual)

        # move into constructor
        self.nstep = 25 # TODO: experiment to find optimal nstep or feed through as parameter
        self.dt = 1/self.nstep

        self.taped = False

        self.tape = fda.Tape()

    def restrict(self, residual, out):
        """
        Restrict from self.V_r_dual into ControlSpace

        Input:
        residual: fd.Cofunction, is a variable in self.V_r_dual
        out: ControlVector, is a variable in the dual of ControlSpace
             (overwritten with result)
        """
        
        old_tape = fda.get_working_tape()
        old_annotation = fda.annotate_tape()
        fda.set_working_tape(self.tape)

        # TODO: Test if below should be commented or not
        self.p0.assign(out.fun) # not sure if this is needed as running forward inside the not taped section?
        # self.residual.assign(residual) # not sure if this is needed also

        if not self.taped:
            fda.continue_annotation()

            self.run_forward()
            # Jhh = fd.assemble(self.residual(self.dphi)) # derivative of this wrt phi is residual, we want derivative of residual wrt to p0 
            self.Jhh_hat = fda.ReducedFunctional(self.dphi, [fda.Control(self.p0)]) # misnomer now can return any overloaded type, functional does not need to be an adj float

            fda.pause_annotation()

            self.taped = True

        self.Jhh_hat([self.p0])
        k = self.Jhh_hat.derivative(adj_input=residual)[0]
        # ic(k.dat.data.min(), k.dat.data.max())
        # k.dat.copy(out.data.dat)
        k.dat.copy(out.cofun.dat)
        # out.cofun.assign(self.Jhh_hat.derivative(adj_input=self.residual)[0])

        # with self.Jhh_hat.derivative(adj_input=residual)[0].dat.vec_ro as v:
        #     v.copy(out.data.dat.vec_wo())
        # with self.Jhh_hat.derivative(adj_input=residual) as derivative:
        #     derivative.copy(out.vec_wo())
        #     # out.cofun.assign(self.Jhh_hat.derivative(adj_input=residual)[0])
        # out.cofun.assign(fd.assemble(Jhh_hat.derivative()[0]))

        fda.set_working_tape(old_tape)
        if old_annotation:
            fda.continue_annotation()
        else:
            fda.pause_annotation()

        # self.tape.visualise("test.dot")

    def interpolate(self, vector, out):
        """
        Interpolate from ControlSpace into self.V_r

        Input:
        vector: ControlVector, is a variable in ControlSpace
        out: fd.Function, is a variable in self.V_r, is overwritten with
             the result
        """
        self.p0.assign(vector.fun)
        self.run_forward()
        out.assign(self.dphi)

    def run_forward(self):
        for _ in range(self.nstep):
            fd.solve(self.a == self.L, self.u0, bcs=self.bcs)
            # ic(self.u0.dat.data.min(), self.u0.dat.data.max())
            self.dphi.interpolate(self.dphi + self.dt * self.u0)

    def get_zero_vec(self):
        fun = fd.Function(self.P)
        fun *= 0.
        return fun

    def get_zero_covec(self):
        fun = fd.Cofunction(self.P_dual)
        fun *= 0.
        return fun

    def get_space_for_inner(self):
        return (self.P, None)

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


class FeControlSpace(ControlSpace):
    """
    Use Lagrangian finite elements as ControlSpace.

    The default option is to use self.V_r as actual ControlSpace.
    If self.V_r is discontinous (e.g. periodic meshes), then create
    a continuous subspace self.V_c (with the same polynomial degree)
    to be used as ControlSpace (to ensure domain updates do not create
    holes in the domain).

    The user can increase the polynomial degree of mesh_r by setting
    the variable add_to_degree_r.

    If mesh_c and degree_c are provided, then use a degree_c Lagrangian
    finite element space defined on mesh_d as ControlSpace.
    Note: as of 15.11.2023, higher-order meshes are not supported, that is,
    mesh_d has to be a polygonal mesh (mesh_r can still be of higher-order).
    """

    def __init__(self, mesh_r, add_to_degree_r=0, mesh_c=None, degree_c=None):
        # Create mesh_r and V_r
        self.mesh_r = mesh_r
        element = self.mesh_r.coordinates.function_space().ufl_element()
        family = element.family()
        degree = element.degree() + add_to_degree_r
        self.V_r = fd.VectorFunctionSpace(self.mesh_r, family, degree)
        self.V_r_dual = self.V_r.dual()

        # Create self.id and self.T, self.mesh_m, and self.V_m.
        X = fd.SpatialCoordinate(self.mesh_r)
        self.id = fd.interpolate(X, self.V_r)
        self.T = fd.Function(self.V_r, name="T")
        self.T.assign(self.id)
        self.mesh_m = fd.Mesh(self.T)
        self.V_m = fd.VectorFunctionSpace(self.mesh_m, family, degree)
        self.V_m_dual = self.V_m.dual()

        if mesh_c is not None:
            if degree_c is None:
                raise ValueError('You need to specify degree_c')
            self.is_decoupled = True
            # Create decoupled FE-control space of mesh_c
            self.V_c = fd.VectorFunctionSpace(mesh_c, "CG", degree_c)
            self.V_c_dual = self.V_c.dual()
            testfct_V_c = fd.TestFunction(self.V_c)
            # Create interpolator from V_c into V_r
            self.Ip = fd.Interpolator(testfct_V_c, self.V_r,
                                      allow_missing_dofs=True)
        elif element.family() == 'Discontinuous Lagrange':
            self.is_DG = True
            self.V_c = fd.VectorFunctionSpace(self.mesh_r, "CG", degree)
            self.V_c_dual = self.V_c.dual()
            testfct_V_c = fd.TestFunction(self.V_c)
            # Create interpolator from V_c into V_r
            self.Ip = fd.Interpolator(testfct_V_c, self.V_r)

    def restrict(self, residual, out):
        if getattr(self, "is_DG", False):
            self.Ip.interpolate(residual, output=out.cofun, transpose=True)
        elif getattr(self, "is_decoupled", False):
            # it's not clear whether this is 100% correct for missing vals
            self.Ip.interpolate(residual, output=out.cofun, transpose=True)
        else:
            out.cofun.assign(residual)

    def interpolate(self, vector, out):
        if getattr(self, "is_DG", False):
            self.Ip.interpolate(vector.fun, output=out)
        elif getattr(self, "is_decoupled", False):
            # extend by zero
            self.Ip.interpolate(vector.fun, output=out, default_missing_val=0.)
        else:
            out.assign(vector.fun)

    def get_zero_vec(self):
        if hasattr(self, "Ip"):
            fun = fd.Function(self.V_c)
        else:
            fun = fd.Function(self.V_r)
        fun *= 0.
        return fun

    def get_zero_covec(self):
        if hasattr(self, "Ip"):
            fun = fd.Cofunction(self.V_c_dual)
        else:
            fun = fd.Cofunction(self.V_r_dual)
        fun *= 0.
        return fun

    def get_space_for_inner(self):
        if hasattr(self, "Ip"):
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
    space. Then, create a finer mesh using `refinements`-many uniform
    refinements and construct a representative of ControlVector that is
    compatible with the state space.

    Inputs:
        refinements: type int, number of uniform refinements to perform
                     to obtain the StateSpace mesh.
        degree: type int, degree of Lagrange basis functions of ControlSpace.

    Note: as of 15.11.2023, higher-order meshes are not supported, that is,
    mesh_r has to be a polygonal mesh (mesh_m can still be of higher-order).
    """

    def __init__(self, mesh_r, refinements=1, degree=1):
        # as of 15.11.2023, MeshHierarchy does not work if
        # refinements_per_level is not a power of 2
        n = refinements
        n_is_power_of_2 = (n & (n-1) == 0) and n != 0
        if not n_is_power_of_2:
            raise NotImplementedError("refinements must be a power of 2")

        # one refinement level with `refinements`-many uniform refinements
        mh = fd.MeshHierarchy(mesh_r, 1, refinements)

        # Control space on coarsest mesh
        self.V_r_coarse = fd.VectorFunctionSpace(mh[0], "CG", degree)
        self.V_r_coarse_dual = self.V_r_coarse.dual()

        # Control space on refined mesh.
        self.mesh_r = mh[-1]
        element = self.V_r_coarse.ufl_element()
        self.V_r = fd.FunctionSpace(self.mesh_r, element)
        self.V_r_dual = self.V_r.dual()

        # Create self.id and self.T on refined mesh.
        X = fd.SpatialCoordinate(self.mesh_r)
        self.id = fd.Function(self.V_r).interpolate(X)
        self.T = fd.Function(self.V_r, name="T")
        self.T.assign(self.id)
        self.mesh_m = fd.Mesh(self.T)
        self.V_m = fd.FunctionSpace(self.mesh_m, element)
        self.V_m_dual = self.V_m.dual()

    def restrict(self, residual, out):
        fd.restrict(residual, out.cofun)

    def interpolate(self, vector, out):
        fd.prolong(vector.fun, out)

    def get_zero_vec(self):
        fun = fd.Function(self.V_r_coarse)
        fun *= 0.
        return fun

    def get_zero_covec(self):
        fun = fd.Cofunction(self.V_r_coarse_dual)
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
        self.V_r_dual = self.V_r.dual()
        X = fd.SpatialCoordinate(self.mesh_r)
        self.id = fd.Function(self.V_r).interpolate(X)
        self.T = fd.Function(self.V_r, name="T")
        self.T.assign(self.id)
        self.mesh_m = fd.Mesh(self.T)
        self.V_m = fd.FunctionSpace(self.mesh_m, element)
        self.V_m_dual = self.V_m.dual()

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

        # lenght of output vector
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
            nnz_ = len(interp_1d[0].getRow(row)[0])  # lenght of nnz-array
            self.IFWnnz = max(self.IFWnnz, nnz_**self.dim)
        IFW.setPreallocationNNZ(self.IFWnnz)
        IFW.setUp()

        for row in range(lsize):
            row = self.lg_map_fe.apply([row])[0]
            M = [[A.getRow(row)[0],
                  A.getRow(row)[1], A.getSize()[1]] for A in interp_1d]
            M = reduce(self.vectorkron, M)
            columns, values, lenght = M
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
            self.cofun = controlspace.get_zero_covec()
        else:
            self.fun = None
            self.cofun = None

    def from_first_derivative(self, fe_deriv):
        if self.boundary_extension is not None:
            if getattr(self.controlspace, "Ip", False):
                raise NotImplementedError("boundary_extension is not"
                      + " supported for discontinous/decoupled meshes") # noqa

            # deep-copy value of fe_deriv
            residual_smoothed = fe_deriv.copy(deepcopy=True)
            # Elasticity-lift -fe_deriv with homogeneous DirBC
            p1 = fe_deriv
            p1 *= -1
            p1_lifted = fd.Function(self.boundary_extension.V)
            self.boundary_extension.solve_homogeneous_adjoint(
                p1, p1_lifted)
            # evaluate elasticity-eqn-form with trialfct=p1_lifted (no BCs)
            self.boundary_extension.apply_adjoint_action(
                p1_lifted, residual_smoothed)
            # add correction to residual_smoothed
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
        if self.cofun is not None:
            self.cofun += v.cofun

    def scale(self, alpha):
        vec = self.vec_wo()
        vec *= alpha
        if self.cofun is not None:
            self.cofun *= alpha

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
