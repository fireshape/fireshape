import firedrake as fd
import fireshape as fs


# create objective functional
class LevelsetFunctional(fs.ShapeObjective):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        x, y = fd.SpatialCoordinate(self.Q.mesh_m)
        self.f = (x - 0.5)**2 + (y - 0.5)**2 - 0.5

    def value_form(self):
        return self.f * fd.dx


# setup problem
mesh = fd.UnitSquareMesh(30, 30)
Q = fs.FeControlSpace(mesh)
q = fs.ControlVector(Q, fs.H1InnerProduct(Q))

# instantiate objective
out = fd.VTKFile("domain_steepest_descent.pvd")
J = LevelsetFunctional(Q, cb=lambda: out.write(Q.mesh_m.coordinates))
J.cb()  # store initial domain once

# manual steepest descent with fixed step size
# weird function signature due to ROL
dq = q.clone()  # deep copy to store gradient
for ii in range(10):
    # evaluate objective function
    print("J(ii =", ii, ") =", J.value(None, None))

    # compute gradient
    J.gradient(dq, None, None)

    # update domain with step size -0.5
    q.axpy(-0.5, dq)
    J.update(q, None, ii)
print("J( final ) =", J.value(None, None))
