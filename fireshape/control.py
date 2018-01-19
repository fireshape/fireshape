# import ROL
import _ROL as ROL
import firedrake as fd

__all__ = ["FeControlSpace", "FeMultiGridControlSpace", "ControlVector"]
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

class FeMultiGridControlSpace(ControlSpace):

    def __init__(self, mesh_r, inner_product):
        super().__init__(mesh_r, inner_product)
        self.mesh_hierarchy = fd.MeshHierarchy(mesh_r, 1, refinements_per_level=3)
        self.mesh_r_coarse = self.mesh_hierarchy[0]
        self.mesh_r_fine = self.mesh_hierarchy[1]
        self.V_r_coarse = self.mesh_r_coarse.coordinates.function_space()

        # self.V_r_fine = self.mesh_r_fine.coordinates.function_space()
        self.V_r_fine = fd.FunctionSpace(self.mesh_r_fine, self.V_r_coarse.ufl_element())

        self.V_r = self.V_r_fine

    def restrict(self, residual):
        fun = fd.Function(self.V_r_coarse)
        fd.restrict(residual, fun)
        return ControlVector(self, data=fun)
    
    def interpolate(self, vector, out):
        vector.copy(out)

    def get_zero_vec(self):
        fun = fd.Function(self.V_r_coarse)
        fun *= 0.
        return fun

    def moving_mesh(self):
        self.id_fine = fd.Function(self.V_r_fine).interpolate(fd.SpatialCoordinate(self.mesh_r_fine))
        self.T_fine = self.id_fine.copy(deepcopy=True)
        self.mesh_m_fine = fd.Mesh(self.T_fine)
        element = self.mesh_m_fine.coordinates.function_space().ufl_element()
        self.V_m_fine = fd.FunctionSpace(self.mesh_m_fine, element)
        return self.mesh_m_fine

    def update_mesh(self, control_vector):
        fd.prolong(control_vector.fun, self.T_fine)
        self.T_fine += self.id_fine



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

