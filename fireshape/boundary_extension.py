import firedrake as fd


class ElasticityExtension(object):

    def __init__(self, V, fixed_dims=[]):
        if isinstance(fixed_dims, int):
            fixed_dims = [fixed_dims]
        self.V = V
        self.fixed_dims = fixed_dims
        self.zero = fd.Constant(V.mesh().topological_dimension() * (0,))

    def extend(self, bc_val, out):
        if len(self.fixed_dims) == 0:
            bcs = [fd.DirichletBC(self.V, bc_val, "on_boundary")]
        else:
            bcs = []
            for i in range(self.V.mesh().topological_dimension()):
                if i in self.fixed_dims:
                    bcs.append(fd.DirichletBC(self.V.sub(i), 0, "on_boundary"))
                else:
                    bcs.append(fd.DirichletBC(self.V.sub(i), bc_val.sub(i), "on_boundary"))

        F = self.get_weak_form(out)
        fd.solve(F == 0, out, bcs=bcs)

    def solve_homogeneous_adjoint(self, rhs, out):
        for i in self.fixed_dims:
            temp = rhs.sub(i)
            temp *= 0
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
        F = 1e-2 * fd.inner(u, v) * fd.dx + fd.inner(fd.sym(fd.grad(u)), fd.grad(v)) * fd.dx
        return F
