# import ROL
import _ROL as ROL
import firedrake as fd

#new imports for splines
from petsc4py import PETSc
# from firedrake.petsc import PETSc
from functools import reduce
from scipy.interpolate import splev
import numpy as np
from .innerproduct import InterpolatedInnerProduct
import ipdb

__all__ = ["BsplineControlSpace", "FeControlSpace", "ControlVector"]
class ControlSpace(object):

    def __init__(self, mesh_r, inner_product):
        self.mesh_r = mesh_r
        self.inner_product = inner_product
        self.V_r = mesh_r.coordinates.function_space()
        self.id = fd.interpolate(fd.SpatialCoordinate(self.mesh_r), self.V_r)

    def restrict(self, residual):
        raise NotImplementedError

    def interpolate(self, vector, out):
        # interpolate vector into fe space and overwrite out with result
        raise NotImplementedError

    def get_zero_vec(self):
        raise NotImplementedError


class FeControlSpace(ControlSpace):

    def __init__(self, mesh_r, inner_product):
        super().__init__(mesh_r, inner_product)

    def restrict(self, residual):
        return ControlVector(self, data=residual)

    def interpolate(self, vector, out):
        vector.copy(out)

    def get_zero_vec(self):
        fun = fd.Function(self.V_r)
        fun *= 0.
        return fun


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
        super().__init__(mesh, inner_product)
        # information on B-splines
        self.dim = len(bbox)
        self.bbox = bbox
        self.orders = orders
        self.levels = levels

        #methods
        self.construct_knots()
        #self.initialize_B()

        self.build_interpolation_matrix()
        self.inner_product = InterpolatedInnerProduct(inner_product, self.interpolate, self.restrict)

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
        self.IFW = self.construct_kronecker_matrix(interp_1d)
        self.construct_index_sets()

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

    def construct_index_sets(self):
        """
        Construct index sets to pull out the (x, y, z)-components of a
        Function defined on self.W.
        """
        self.ises = [] #index sets for PETSc control vector
        for dim in range(self.dim):
            dofs_dim = np.array(list(range(self.N)), dtype=np.int32) + dim*self.N
            is_ = PETSc.IS().createGeneral(dofs_dim, comm=self.mesh_r.mpi_comm())
            self.ises.append(is_)

        self.isesFD = [] #index sets for firedrake function
        for dim in range(self.dim):
            #index are interleaved
            dofs_dim = self.dim*np.array(list(range(self.M)), dtype=np.int32) + dim
            is_ = PETSc.IS().createGeneral(dofs_dim, comm=self.mesh_r.mpi_comm())
            # Possible FIXME: it's much faster to use IS().createStride()
            self.isesFD.append(is_)

    def restrict(self, residual):
        """
        Takes in a Function in self.V_r. Returns a PETSc vector of length self.N*self.dim.

        Used to approximate the evaluation of a linear functional (on-
        cartesian multivariate B-splines) with linear combinations of
        values of the same linear functional on basis functions in self.W .
        """
        #shall we use createMPI?
        out =  PETSc.Vec().createSeq(self.N*self.dim, comm=self.mesh_r.mpi_comm())
        with residual.dat.vec as w:
            for dim in range(self.dim):
                newvalues = PETSc.Vec().createSeq(self.N, comm=self.mesh_r.mpi_comm())
                wpart = w.getSubVector(self.isesFD[dim])
                if not self.IFW.size[0] == wpart.size:
                    ipdb.set_trace()
                self.IFW.multTranspose(wpart ,newvalues)
                w.restoreSubVector(self.isesFD[dim], wpart)
                out.isaxpy(self.ises[dim], 1.0, newvalues)
        return out


    def interpolate(self, vector, out):
        """
        Takes in the coefficients of a vector field in spline space.
        Returns its interpolant in self.V_r

        vector is a PETSc vector of length self.N*self.dim
        """

        # Construct the Function in self.W.
        #with out.dat.vec as w:
        for dim in range(self.dim):
            newvalues = PETSc.Vec().createSeq(self.M, comm=self.mesh_r.mpi_comm())
            vectorpart = vector.vec.getSubVector(self.ises[dim])
            self.IFW.mult(vectorpart, newvalues)
            vectorpart = vector.vec.restoreSubVector(self.ises[dim], vectorpart)
            print(type(out))
            #if isinstance(out,  fd.CoordinatelessFunction):
            #    ipdb.set_trace()
            #with out.vec as outvec:
            #    outvec.isaxpy(self.isesFD[dim], 1.0, newvalues)
            #else:
            out.isaxpy(self.isesFD[dim], 1.0, newvalues)

    def get_zero_vec(self):
        vec = PETSc.Vec().createSeq(self.N*self.dim, comm=self.mesh_r.mpi_comm())
        return ControlVector(BsplineControlSpace, data=vec)

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

        self.Tfem = None
        self.Omega = None
        self.V_m = None

    def plus(self, v):
        self.vec += v.vec

    def scale(self, alpha):
        self.vec *= alpha

    def clone(self): # misleading name from ROL, returns a vector
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

    def domain(self):
        self.update_domain()
        if self.Omega is None:
            self.Omega = fd.Mesh(self.Tfem)
        return self.Omega

    def update_domain(self):
        if self.Tfem is None:
            self.Tfem = fd.Function(self.controlspace.V_r)

        with self.Tfem.dat.vec as v:
            self.controlspace.interpolate(self.vec, v)
        self.Tfem += self.controlspace.id

    def V(self):
        if self.V_m is None:
            mesh_m = self.domain()
            element = mesh_m.coordinates.function_space().ufl_element()
            self.V_m = fd.FunctionSpace(mesh_m, element)
        return self.V_m

    def __str__(self):
        return self.vec[:].__str__()

