import ROL
import firedrake as fd

class Control(ROL.Vector):
    def __init__(self, mesh, inner_product):
        self.mesh = mesh
        V = mesh.coordinates.function_space()
        self.Tfem = fd.interpolate(fd.SpatialCoordinate(self.mesh), V)
        self.inner_product = inner_product

    def domain():
        if self.Omega is None:
            self.Omega = fd.Mesh(self.Tfem)
        return self.Omega

    def update_domain():
        raise NotImplementedError()

    def plus(self, v):
        raise NotImplementedError()

    def scale(self, alpha):
        raise NotImplementedError()

    def clone(self):
        raise NotImplementedError()

    def inner(self, v):
        return self.inner_product.eval(self, v)


class FemControl(Control):
    def __init__(self, mesh, inner_product):
        pass

    def update_domain(self):
        pass


class BsplineControl(ROL.Vector):

    def __init__(self, mesh, inner_product):
        self.mesh = mesh

        # ...
        self.interp = #...
        self.inner_product = InterpolatingInnerProduct(inner_product, interp)
        self.vec = ... 

    def update_domain(self):
        # Tfem= id + Ih*self.vec
        # mult with interp
