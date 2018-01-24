# import ROL
import _ROL as ROL
import firedrake as fd

__all__ = ["FeControlSpace", "FeMultiGridControlSpace", "ControlVector"]
class ControlSpace(object):

    def update_domain(self, q: 'ControlVector'):
        raise NotImplementedError

    def restrict(self, residual):
        raise NotImplementedError
    
    def interpolate(self, vector, out):
        # interpolate vector into fe space and overwrite out with result
        raise NotImplementedError

    def get_zero_vec(self):
        raise NotImplementedError


class FeControlSpace(ControlSpace):

    def __init__(self, mesh_r, inner_product):
        self.mesh_r = mesh_r
        self.inner_product = inner_product
        self.V_r = mesh_r.coordinates.function_space()
        self.id = fd.interpolate(fd.SpatialCoordinate(self.mesh_r), self.V_r)
        self.T = fd.Function(self.V_r, name="T")
        self.T.assign(self.id)
        self.mesh_m = fd.Mesh(self.T)
        element = self.mesh_m.coordinates.function_space().ufl_element()
        self.V_m = fd.FunctionSpace(self.mesh_m, element)

    def update_domain(self, q: 'ControlVector'):
        with self.T.dat.vec as v:
            self.interpolate(q.vec, v)
        self.T+= self.id

    def restrict(self, residual):
        return ControlVector(self, data=residual)
    
    def interpolate(self, vector, out):
        vector.copy(out)

    def get_zero_vec(self):
        fun = fd.Function(self.V_r)
        fun *= 0.
        return fun

class FeMultiGridControlSpace(ControlSpace):

    def __init__(self, mesh_r, inner_product, refinements_per_level=1):
        self.inner_product = inner_product
        self.mesh_hierarchy = fd.MeshHierarchy(mesh_r, 1, refinements_per_level=refinements_per_level)
        self.mesh_r_coarse = self.mesh_hierarchy[0]
        self.V_r_coarse = self.mesh_r_coarse.coordinates.function_space()
        self.mesh_r = self.mesh_hierarchy[1]
        self.V_r = fd.FunctionSpace(self.mesh_r, self.V_r_coarse.ufl_element())
        self.id = fd.Function(self.V_r).interpolate(fd.SpatialCoordinate(self.mesh_r))
        self.T = fd.Function(self.V_r, name="T")
        self.T.assign(self.id)
        self.mesh_m = fd.Mesh(self.T)
        element = self.mesh_m.coordinates.function_space().ufl_element()
        self.V_m = fd.FunctionSpace(self.mesh_m, element)

    def restrict(self, residual):
        fun = fd.Function(self.V_r_coarse)
        fd.restrict(residual, fun)
        return ControlVector(self, data=fun)
    
    def interpolate(self, vector, out):
        fd.prolong(vector.fun, out)

    def get_zero_vec(self):
        fun = fd.Function(self.V_r_coarse)
        fun *= 0.
        return fun

    def update_domain(self, q: 'ControlVector'):
        self.interpolate(q, self.T)
        self.T += self.id



class BsplineControlSpace(ControlSpace):

    def __init__(self, mesh, inner_product):
        super().__init__(mesh, inner_product)
        self.interp = self.build_interpolation_matrix()

    def build_interpolation_matrix(self):
        # TODO
        raise NotImplementedError

    def restrict(self, residual):
        # self.interp.T * residual
        raise NotImplementedError
    
    def interpolate(self, vector, out):
        # self.interp * vector
        raise NotImplementedError

    def get_zero_vec(self):
        # new petsc vec ...
        # return vec
        raise NotImplementedError

class ControlVector(ROL.Vector):

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
        # self.fun is the firedrake function object wrapping around
        # the petsc vector self.vec. If the vector does not correspond
        # to a firedrake function, then self.fun is None

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

    def __str__(self):
        return self.vec[:].__str__()

