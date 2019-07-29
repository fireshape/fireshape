import firedrake as fd


class BilinearForm():

    def get_form(self, V):
        raise NotImplementedError

    def is_symmetric(self):
        raise NotImplementedError


class LaplaceForm(BilinearForm):

    def __init__(self, *args, mu=fd.Constant(1.0), **kwargs):
        super().__init__(*args, **kwargs)
        self.mu = mu

    def get_form(self, V):
        u = fd.TrialFunction(V)
        v = fd.TestFunction(V)
        return self.mu * fd.inner(fd.grad(u), fd.grad(v)) * fd.dx

    def get_nullspace(self, V):
        """This nullspace contains constant functions."""
        dim = V.value_size
        if dim == 2:
            n1 = fd.Function(V).interpolate(fd.Constant((1.0, 0.0)))
            n2 = fd.Function(V).interpolate(fd.Constant((0.0, 1.0)))
            res = [n1, n2]
        elif dim == 3:
            n1 = fd.Function(V).interpolate(fd.Constant((1.0, 0.0, 0.0)))
            n2 = fd.Function(V).interpolate(fd.Constant((0.0, 1.0, 0.0)))
            n3 = fd.Function(V).interpolate(fd.Constant((0.0, 0.0, 1.0)))
            res = [n1, n2, n3]
        else:
            raise NotImplementedError
        return res

    def is_symmetric(self):
        return True


class ElasticityForm(BilinearForm):

    def __init__(self, *args, mu=fd.Constant(1.0), **kwargs):
        super().__init__(*args, **kwargs)
        self.mu = mu

    def get_form(self, V):
        u = fd.TrialFunction(V)
        v = fd.TestFunction(V)
        symgrad = lambda x: fd.sym(fd.grad(x))  # noqa
        a = self.mu * fd.inner(symgrad(u), symgrad(v)) * fd.dx
        return a

    def get_nullspace(self, V):
        """
        This nullspace contains constant functions as well as rotations.
        """
        X = fd.SpatialCoordinate(V.mesh())
        dim = V.value_size
        if dim == 2:
            n1 = fd.Function(V).interpolate(fd.Constant((1.0, 0.0)))
            n2 = fd.Function(V).interpolate(fd.Constant((0.0, 1.0)))
            n3 = fd.Function(V).interpolate(fd.as_vector([X[1], -X[0]]))
            res = [n1, n2, n3]
        elif dim == 3:
            n1 = fd.Function(V).interpolate(fd.Constant((1.0, 0.0, 0.0)))
            n2 = fd.Function(V).interpolate(fd.Constant((0.0, 1.0, 0.0)))
            n3 = fd.Function(V).interpolate(fd.Constant((0.0, 0.0, 1.0)))
            n4 = fd.Function(V).interpolate(fd.as_vector([-X[1], X[0], 0]))
            n5 = fd.Function(V).interpolate(fd.as_vector([-X[2], 0, X[0]]))
            n6 = fd.Function(V).interpolate(fd.as_vector([0, -X[2], X[1]]))
            res = [n1, n2, n3, n4, n5, n6]
        else:
            raise NotImplementedError
        return res

    def is_symmetric(self):
        return True


class GoodHelmholtzForm(BilinearForm):

    def __init__(self, *args, mu=fd.Constant(1.0), mu_grad=fd.Constant(1.0),
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.mu = mu
        self.mu_grad = mu_grad

    def get_form(self, V):
        u = fd.TrialFunction(V)
        v = fd.TestFunction(V)
        return self.mu * fd.inner(u, v) * fd.dx \
            + self.mu_grad * fd.inner(fd.grad(u), fd.grad(v)) * fd.dx

    def get_nullspace(self, V):
        return None

    def is_symmetric(self):
        return True


class CauchyRiemannAugmentation(BilinearForm):

    def __init__(self, base_form, *args, mu=fd.Constant(1.0), **kwargs):
        super().__init__(*args, **kwargs)
        self.base_form = base_form
        self.mu = mu

    def get_form(self, V):
        assert V.value_size == 2

        def cr_op(u):
            return fd.as_vector([fd.grad(u[0])[0]-fd.grad(u[1])[1],
                                 fd.grad(u[0])[1] + fd.grad(u[1])[0]])
        u = fd.TrialFunction(V)
        v = fd.TestFunction(V)
        return self.mu * fd.inner(cr_op(u), cr_op(v)) * fd.dx \
            + self.base_form.get_form(V)

    def get_nullspace(self, V):
        return self.base_form.get_nullspace(V)

    def is_symmetric(self):
        return True
