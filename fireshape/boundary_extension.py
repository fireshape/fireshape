import firedrake as fd


class ElasticityExtension(object):

    def __init__(self, V):
        self.V = V
        self.zero = fd.Constant(V.mesh().topological_dimension() * (0,))

    def extend(self, bc_val, out):
        bc = fd.DirichletBC(self.V, bc_val, "on_boundary")
        v = fd.TestFunction(self.V)
        F = fd.inner(fd.sym(fd.grad(out)), fd.grad(v)) * fd.dx
        fd.solve(F == 0, out, bcs=bc)

    def solve_homogeneous_adjoint(self, rhs, out):
        bc = fd.DirichletBC(self.V, self.zero, "on_boundary")
        F = self.get_weak_form(out)
        a = fd.adjoint(fd.derivative(F, out))
        fd.solve(fd.assemble(a), out, rhs, bcs=bc)

    def apply_adjoint_action(self, x, out):
        F = self.get_weak_form(out)
        a = fd.adjoint(fd.derivative(F, out))
        out.assign(fd.assemble(fd.action(a, x)))

    def get_weak_form(self, u):
        v = fd.TestFunction(self.V)
        F = fd.inner(fd.sym(fd.grad(u)), fd.grad(v)) * fd.dx
        return F
