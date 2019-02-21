import firedrake as fd
import fireshape as fs

class LevelsetFunctional(fs.ShapeObjective):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        #physical mesh
        self.mesh_m = self.Q.mesh_m

        #global function defined in terms of physical coordinates
        x,y = fd.SpatialCoordinate(self.mesh_m)
        self.f = (x - 0.5)**2 + (y - 0.5)**2 - 0.5

    def value_form(self):
        #volume integral
        return self.f * fd.dx
