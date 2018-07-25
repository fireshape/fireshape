import firedrake as fd
import fireshape as fs

class LevelsetFunctional(fs.ShapeObjective):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        #physical mesh
        mesh_m = self.Q.mesh_m

        #global function defined in terms of physical coordinates
        x,y = fd.SpatialCoordinate(mesh_m)
        self.f = (x*x-1)*(y*y-1)

    def value_form(self):
        #volume integral
        return self.f * fd.dx

    def derivative_form(self, v):
        #shape differentiate J in the direction v
        X = SpatialCoordinate(self.mesh)
        return derivative(value_form, X, v)
