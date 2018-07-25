import firedrake as fd
import fireshape as fs

class LevelsetFunctional(fs.ShapeObjective):
    """
    Implementation of level-set shape functional.

    Optima are zero-levels of ufl function f.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        #physical mesh
        mesh_m = self.Q.mesh_m

        #global function defined in terms of coordinates
        #from the physical space
        x,y = fd.SpatialCoordinate(mesh_m)
        self.f = (x*x-1)*(y*y-1)

    def value_form(self):
        #volume integral
        return self.f * fd.dx

    def derivative_form(self, v):
        return fd.div(self.f*v) * fd.dx
