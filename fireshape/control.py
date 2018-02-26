import ROL
import firedrake as fd

__all__ = ["FeControlSpace", "FeMultiGridControlSpace", "BsplineControlSpace", "ControlVector"]


#new imports for splines
from firedrake.petsc import PETSc
from functools import reduce
from scipy.interpolate import splev
import numpy as np
from .innerproduct import InterpolatedInnerProduct

class ControlSpace(object):
    """
    ControlSpace is the space of geometric transformations.

    A transformation is identified with a domain using Firedrake. In particular,
    a transformation is converted into a domain by interpolating it on a
    Firedrake Lagrangian finite element space.

    Notational convention:
        self.mesh_r is the initial physical mesh (reference domain)
        self.V_r is the Firedrake vectorial Lagrangian finite element space on mesh_r
        self.id is the element of V_r that satisfies id(x) = x for every x
        self.T is the interpolant of this ControlSpace variable in self.V_r
        self.mesh_m is the mesh that corresponds to self.T (moved domain)
        self.V_m is the Firedrake vectorial Lagrangian finite element space on mesh_m
        self.inner_product is the inner product of the ControlSpace

    """
    def update_domain(self, q: 'ControlVector'):
        """
        Update the interpolant self.T

        shall we implement this here?
        with self.T.dat.vec as v:
            self.interpolate(q.vec, v)
        self.T += self.id
        why is FeMultiGridControlSpace.update_domain different?
        """
        raise NotImplementedError

    def restrict(self, residual, out):
        """
        Restrict from self.V_r into ControlSpace
        Input: residual is a variable in self.V_r
               out is a variable in ControlSpace, is overwritten with the result
            (if we modify FeMultiGridControlSpace.restrict, then we can write that
            out.vec is overwritten with the result)
        """
        raise NotImplementedError

    def interpolate(self, vector, out):
        """
        Interpolate from ControlSpace into self.V_r
        Input: vector is a variable in ControlSpace
               out is a variable in self.V_r, is overwritten with the result
        """
        raise NotImplementedError

    def get_zero_vec(self):
        """
        Create a variable in ControlSpace the corresponds to zero
        """
        raise NotImplementedError


class FeControlSpace(ControlSpace):

    def __init__(self, mesh_r, inner_product):
        self.mesh_r = mesh_r
        element = self.mesh_r.coordinates.function_space().ufl_element()
        self.V_r = fd.FunctionSpace(self.mesh_r, element)
        #FeMultiGridControlSpace and BsplineControlSpace use a differen
        #construction of self.id
        self.id = fd.interpolate(fd.SpatialCoordinate(self.mesh_r), self.V_r)
        self.T = fd.Function(self.V_r, name="T")
        self.T.assign(self.id)
        self.mesh_m = fd.Mesh(self.T)
        self.V_m = fd.FunctionSpace(self.mesh_m, element)
        self.inner_product = inner_product.get_impl(self.V_r)

    def update_domain(self, q: 'ControlVector'):
        with self.T.dat.vec as v:
            self.interpolate(q.vec, v)
        self.T += self.id

    def restrict(self, residual, out):
        with residual.dat.vec as vecres:
            vecres.copy(out.vec)

    def interpolate(self, vector, out):
        vector.copy(out)

    def get_zero_vec(self):
        fun = fd.Function(self.V_r)
        fun *= 0.
        return fun


class FeMultiGridControlSpace(ControlSpace):

    def __init__(self, mesh_r, inner_product, refinements=1, order=1):
        mh = fd.MeshHierarchy(mesh_r, 1, refinements_per_level=refinements)
        self.mesh_hierarchy = mh
        self.mesh_r_coarse = self.mesh_hierarchy[0]
        self.V_r_coarse = fd.VectorFunctionSpace(self.mesh_r_coarse, "CG", order)

        self.mesh_r = self.mesh_hierarchy[1]
        element = self.V_r_coarse.ufl_element()
        self.V_r = fd.FunctionSpace(self.mesh_r, element)
        X = fd.SpatialCoordinate(self.mesh_r)
        self.id = fd.Function(self.V_r).interpolate(X)
        self.T = fd.Function(self.V_r, name="T")
        self.T.assign(self.id)
        self.mesh_m = fd.Mesh(self.T)
        self.V_m = fd.FunctionSpace(self.mesh_m, element)

        self.inner_product = inner_product.get_impl(self.V_r_coarse)

    def restrict(self, residual, out):
        fd.restrict(residual, out.fun) #better if we overwrite out.vec

    def interpolate(self, vector, out):
        fd.prolong(vector.fun, out)

    def get_zero_vec(self):
        fun = fd.Function(self.V_r_coarse)
        fun *= 0.
        return fun

    def update_domain(self, q: 'ControlVector'):
        self.interpolate(q, self.T)#why is it different form the other ones
        self.T += self.id


class BsplineControlSpace(ControlSpace):

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
        self.dim = len(bbox)
        self.bbox = bbox
        self.orders = orders
        self.levels = levels
        self.construct_knots()

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

        # interpolated inner product
        self.build_interpolation_matrix()
        A = inner_product.get_impl(self.V_r).A
        self.inner_product = InterpolatedInnerProduct(A, self.FullIFW)

    def construct_knots(self):
        """
        construct self.knots, self.n, self.NA

        self.knots is a list of np.arrays (one per geometric dimension)
        each array corresponds to the knots used to define the spline space
        """
        self.knots = []
        self.n = []
        for dim in range(self.dim):
            order = self.orders[dim]
            level = self.levels[dim]

            assert order >= 1
            degree = order-1 #splev uses degree, not order
            assert level >= 1 #with level=1 only bdry Bsplines

            knots_01 = np.concatenate((np.zeros((order-1,), dtype=float),
                                       np.linspace(0., 1., 2**level+1),
                                       np.ones((order-1,), dtype=float)))

            (xmin, xmax) = self.bbox[dim]
            knots = (xmax - xmin)*knots_01 + xmin
            self.knots.append(knots)
            #list of dimension of univariate spline spaces
            #the "-2" is because we want homogeneous Dir bc
            n = len(knots) - order - 2
            assert n > 0
            self.n.append(n)

        #dimension of multivariate spline space
        N = reduce(lambda x, y: x*y, self.n)
        self.N = N

    def build_interpolation_matrix(self):
        interp_1d = self.construct_1d_interpolation_matrices()
        IFW = self.construct_kronecker_matrix(interp_1d)
        self.FullIFW = self.construct_full_interpolation_matrix(IFW)

    def construct_1d_interpolation_matrices(self):
        """
        Create a list of sparse matrices (one per geometric dimension).
        Each matrix has size (M, n[dim]), where M is the dimension of the
        FE interpolation space, and n[dim] is the dimension of the univariate
        spline space associated to the dimth-geometric coordinate.
        The ith column of such a matrix is computed by evaluating the ith
        univariate B-spline on the dimth-geometric coordinate of the dofs of
        the FE interpolation space
        """
        interp_1d = []

        x_fct = fd.SpatialCoordinate(self.mesh_r) #used for x_int, replace with self.id
        x_int = fd.interpolate(x_fct[0], self.V_r.sub(0))
        self.M = x_int.vector().size() #no dofs for (scalar) fct in self.V_r.sub(0)
        #import pdb
        #pdb.set_trace()
        for dim in range(self.mesh_r.geometric_dimension()):
            order = self.orders[dim]
            knots = self.knots[dim]
            n = self.n[dim]

            I = PETSc.Mat().create(comm=self.mesh_r.mpi_comm())
            I.setType(PETSc.Mat.Type.AIJ)
            I.setSizes((self.M, n))
            # BIG TODO: figure out the sparsity pattern
            I.setUp()

            #todo: read fecoords out of  self.id, so far
            x_int = fd.interpolate(x_fct[dim], self.V_r.sub(0))
            with x_int.dat.vec_ro as x:
                for idx in range(n):
                    coeffs = np.zeros(knots.shape, dtype=float)
                    coeffs[idx+1] = 1 #idx+1 because we consider hom Dir bc for splines
                    degree = order - 1
                    tck = (knots, coeffs, degree)

                    values = splev(x.array, tck, der=0, ext=1)
                    rows = np.where(values != 0)[0].astype(np.int32)
                    values = values[rows]
                    I.setValues(rows, [idx], values)

            I.assemble()# lazy strategy for kron
            interp_1d.append(I)

        return interp_1d

    def construct_kronecker_matrix(self, interp_1d):
        # this may be done matrix-free

        """ 
        Construct the definitive interpolation matrix by computing
        the kron product of the rows of the 1d univariate interpolation
        matrices
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
        FullIFW = PETSc.Mat().create(self.mesh_r.mpi_comm())
        FullIFW.setType(PETSc.Mat.Type.AIJ)
        FullIFW.setSizes((self.dim * self.M, self.dim * self.N))
        # BIG TODO: figure out the sparsity pattern
        FullIFW.setUp()

        for row in range(self.M):
            (cols, vals) = IFW.getRow(row)
            for dim in range(self.dim):
                FullIFW.setValues([self.dim * row + dim],
                                  [self.dim * col + dim for col in cols],
                                  vals)
        FullIFW.assemble()
        return FullIFW


    def restrict(self, residual, out):
        """
        Takes in a Function in self.V_r. Returns a PETSc vector of length self.N*self.dim.

        Used to approximate the evaluation of a linear functional (on-
        cartesian multivariate B-splines) with linear combinations of
        values of the same linear functional on basis functions in self.W .
        """
        with residual.dat.vec as w:
            self.FullIFW.multTranspose(w, out.vec)

    def interpolate(self, vector, out):
        """
        Takes in the coefficients of a vector field in spline space.
        Returns its interpolant in self.V_r

        vector is a PETSc vector of length self.N*self.dim
        """

        # Construct the Function in self.W.
        self.FullIFW.mult(vector, out)

    def get_zero_vec(self):
        vec = PETSc.Vec().createSeq(self.N*self.dim, comm=self.mesh_r.mpi_comm())
        return vec

    def update_domain(self, q: 'ControlVector'):
        with self.T.dat.vec as w:
            self.interpolate(q.vec, w)
        self.T += self.id

class ControlVector(ROL.Vector):

    def __init__(self, controlspace: ControlSpace, data=None):
        super().__init__()
        self.controlspace = controlspace

        if data is None:
            data = controlspace.get_zero_vec()

        if isinstance(data, fd.Function): #is it a good ide to store the additional info in self.fun?
            self.fun = data
            with data.dat.vec as v:
                self.vec = v
        else:
            self.vec = data #self.vec is always a PETSc vector
            self.fun = None
        # self.fun is the firedrake function object wrapping around
        # the petsc vector self.vec. If the vector does not correspond
        # to a firedrake function, then self.fun is None

    def plus(self, v):
        self.vec += v.vec

    def scale(self, alpha):
        self.vec *= alpha

    def clone(self):
        # misleading name from ROL, returns a vector
        # of same size but containing zeros
        res = ControlVector(self.controlspace)
        # res.set(self)
        return res

    def dot(self, v):
        return self.controlspace.inner_product.eval(self, v)

    def axpy(self, alpha, x):
        self.vec.axpy(alpha, x.vec)

    def set(self, v):
        v.vec.copy(self.vec)

    def __str__(self):
        return self.vec[:].__str__()
