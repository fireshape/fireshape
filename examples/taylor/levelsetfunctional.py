import firedrake as fd
import fireshape as fs
from icecream import ic
ic.configureOutput(includeContext=True) 

class LevelsetFunctional(fs.ShapeObjective):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # physical mesh
        self.mesh_m = self.Q.mesh_m

        # S = fd.FunctionSpace(self.mesh_m, "DG", 0)
        # self.I = fd.Function(S, name="indicator")
        # global function defined in terms of physical coordinates
        x, y = fd.SpatialCoordinate(self.mesh_m)

        # self.I.interpolate(fd.conditional(x < 1, fd.conditional(x > 0, fd.conditional(y > 0, fd.conditional(y < 1, 1, 0), 0), 0), 0))
        self.f = (x - 0.5)**2 + (y - 0.5)**2 - 0.5
        # self.scale = 0.5 # this doesnt do anything
        # self.f = (x + self.Q.dphi[0] - 0.5)**2 + (y + self.Q.dphi[1] - 0.5)**2 - 0.5

    def value_form(self):
        # volume integral
        # TODO: maybe this needs *I idk?
        k = self.f * fd.dx
        # ic(fd.assemble(k))
        return k
