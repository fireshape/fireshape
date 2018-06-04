import fireshape as fs
import firedrake as fd


class CauchyRiemannAugmentation(fs.UflInnerProduct):

    def __init__(self, mu_cr, inner):
        self.inner = inner
        self.mu_cr = mu_cr
        super().__init__(inner.Q, fixed_bids=inner.fixed_bids,
                         extra_bcs=inner.extra_bcs)

    def get_weak_form(self, V):
        """
        mu is a spatially varying coefficient in the weak form
        of the elasticity equations. The idea is to make the mesh stiff near
        the boundary that is being deformed.
        """
        a = self.inner.get_weak_form(V)
        u = fd.TrialFunction(V)
        v = fd.TestFunction(V)
        (u1, u2) = fd.split(u)
        (v1, v2) = fd.split(v)
        a_cr = (fd.inner(fd.grad(u1)[0] - fd.grad(u2)[1],
                         fd.grad(v1)[0] - fd.grad(v2)[1])
                + fd.inner(fd.grad(u1)[1] + fd.grad(u2)[0],
                           fd.grad(v1)[1] + fd.grad(v2)[0])
                ) * self.mu_cr * fd.dx
        return a + a_cr
