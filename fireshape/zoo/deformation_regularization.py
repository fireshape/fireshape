import firedrake as fd
import fireshape as fs


__all__ = ["DeformationRegularization", "CoarseDeformationRegularization"]


class DeformationRegularization(fs.DeformationObjective):

    def __init__(self, *args, l2_reg=1., sym_grad_reg=1., skew_grad_reg=1.,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.T = self.Q.T
        self.id = self.T.copy(deepcopy=True)
        self.id.interpolate(fd.SpatialCoordinate(self.T.ufl_domain()))
        self.l2_reg = l2_reg
        self.sym_grad_reg = sym_grad_reg
        self.skew_grad_reg = skew_grad_reg

    def value_form(self):
        f = self.T-self.id

        def norm(u):
            return fd.inner(u, u) * fd.dx
        val = self.l2_reg * norm(f)
        val += self.sym_grad_reg * norm(fd.sym(fd.grad(f)))
        val += self.skew_grad_reg * norm(fd.skew(fd.grad(f)))
        return val

    def derivative_form(self, test):
        T = self.T
        return fd.derivative(self.value_form(), T, test)


class CoarseDeformationRegularization(fs.ControlObjective):

    def __init__(self, form, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.T = self.Q.intermediate_Ts[0]
        self.form = form.get_form(self.T.function_space())
        self.id = self.Q.intermediate_Ids[0]
        self.temp = self.id.copy(deepcopy=True)

    def value_form(self):
        self.temp.assign(self.T)
        self.temp -= self.id
        f = self.temp
        return fd.action(fd.action(self.form, f), f)

    def derivative_form(self, test):
        self.temp.assign(self.T)
        self.temp -= self.id
        f = self.temp
        deriv = fd.derivative(self.value_form(), f, test)
        return deriv
