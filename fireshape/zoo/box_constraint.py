import firedrake as fd
import fireshape as fs


__all__ = ["MoYoBoxConstraint"]


def Min(a, b):
    return (a+b-abs(a-b))/fd.Constant(2)


def Max(a, b):
    return (a+b+abs(a-b))/fd.Constant(2)


class MoYoBoxConstraint(fs.DeformationObjective):

    def __init__(self, c, bids, *args, lower_bound=None,
                 upper_bound=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.T = self.Q.T
        self.lam = fd.Function(self.T.function_space())
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        self.bids = bids
        self.viol = fd.Function(self.T.function_space())
        self.c = c

    def value_form(self):
        lam = self.lam
        c = self.c
        psi = self.upper_bound
        phi = self.lower_bound
        T = self.T
        temp0 = Max(lam[0] + c*(T[0]-psi[0]), fd.Constant(0.0)) \
            + Min(lam[0] + c*(T[0]-phi[0]), fd.Constant(0.0))
        temp1 = Max(lam[1] + c*(T[1]-psi[1]), fd.Constant(0.0)) \
            + Min(lam[1] + c*(T[1]-phi[1]), fd.Constant(0.0))
        val = (1.0/(2.0*c)) * (
            self.dot(temp0, temp0)+self.dot(temp1, temp1) - self.dot(lam, lam))
        return val

    def derivative_form(self, test):
        T = self.T
        return fd.derivative(self.value_form(), T, test)

    def dot(self, u, v):
        return sum([fd.inner(u, v)*fd.ds(bid) for bid in self.bids])


def RelevantPartOfVector(vec, maximum, compare=0.0):
    relvec = vec.copy()
    comparevec = vec.copy()
    relvec *= 0
    from firedrake import as_backend_type
    pvec = as_backend_type(vec).vec()
    pcomparevec = as_backend_type(comparevec).vec()
    pcomparevec.set(compare)
    prelvec = as_backend_type(relvec).vec()
    if maximum:
        prelvec.pointwiseMax(pcomparevec, pvec)
    else:
        prelvec.pointwiseMin(pcomparevec, pvec)
    return relvec
