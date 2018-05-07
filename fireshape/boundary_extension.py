import firedrake as fd

class ElasticityExtension(object):

    def __init__(self, V):
        self.V = V

    def extend(self, bc_val, out):
        bc = fd.DirichletBC(self.V, bc_val, "on_boundary")
        v = fd.TestFunction(self.V)
        F = fd.inner(fd.sym(fd.grad(out)), fd.grad(v)) * fd.dx
        fd.solve(F==0, out, bcs=bc)

