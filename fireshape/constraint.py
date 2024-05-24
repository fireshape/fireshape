"""
import ROL


__all__ = ["EqualityConstraint"]


class EqualityConstraint(ROL.Constraint):

    def __init__(self, c, target_value=None):
        super().__init__()

        if target_value is None:
            target_value = [c_.value(None, None) for c_ in c]
        self.target_value = target_value
        self.c = c

    def value(self, c, x, tol):
        for i in range(len(self.c)):
            c[i] = self.c[i].value(None, None) - self.target_value[i]

    def applyJacobian(self, jv, v, x, tol):
        g = v.clone()
        for i in range(len(self.c)):
            self.c[i].gradient(g, x, tol)
            jv[i] = g.dot(v)

    def applyAdjointJacobian(self, ajv, v, x, tol):
        ajv.scale(0.0)
        g = ajv.clone()
        for i in range(len(self.c)):
            self.c[i].gradient(g, x, tol)
            g.scale(v[i])
            ajv.plus(g)

    def update(self, x, flag, iteration):
        for c_ in self.c:
            c_.update(x, flag, iteration)
"""
