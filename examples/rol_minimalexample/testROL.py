import ROL
from numpy import *

class ControlVector(ROL.Vector):
    """
    A ControlVector is a variable in the ControlSpace.
    In this example, the data of a control vector is a double
    A ControlVector is a ROL.Vector and thus needs the following methods:
    plus, scale, clone, dot, axpy, set.
    """
    def __init__(self):
        super().__init__()
        #print('initiating', flush=True)
        self.x = 0.
        self.y = 0.

    def plus(self, v):
        #print('adding', flush=True)
        self.x += v.x
        self.y += v.y

    def scale(self, alpha):
        #print('scaling', flush=True)
        self.x *= alpha
        self.y *= alpha

    def clone(self):
        #print('cloning', flush=True)
        return ControlVector()

    def dot(self, v):
        #print('dotting', flush=True)
        return self.y * v.y + self.x * v.x

    def norm(self):
        #print('norming', flush=True)
        return (self.x**2 + self.y**2)**0.5

    def axpy(self, alpha, y):
        self.x = self.x*alpha + y.x
        self.y = self.y*alpha + y.y

    def set(self, v):
        #print('setting', flush=True)
        self.x = v.x
        self.y = v.x

class Objective(ROL.Objective):
    def __init__(self, cb=None):
        super().__init__()
        self.val = [1, 2, 3, nan, nan, 2, 0.8, nan, nan]
        self.iter = 0

    def value(self, x, tol):
        #return (x.x-2)**2 * (x.y-1)**2
        val = self.val[self.iter]
        print('evaluate function', flush=True)
        self.iter += 1
        return val

    def gradient(self, g, x, tol):
        print('evaluate gradient', flush=True)
        g.x = 2#2*(x.x-2) * (x.y-1)**2
        g.y = 1#2*(x.y-1) * (x.x-2)**2

    def update(self, x, flag, iteration):
        #super().update(x, flag, iteration)
        print('updating', flush=True)

class EqualityConstraint(ROL.Constraint):

    def __init__(self, c=None, target_value=None):
        super().__init__()

    def value(self, c, x, tol):
        return x.x**2 + x.y**2 - 1

if __name__== "__main__":
    #print('call ControlVector()', flush=True)
    q = ControlVector()
    #print('call Objective()', flush=True)
    J = Objective()
    econ = EqualityConstraint()
    emul = ROL.StdVector(1)
    params_dict = {
    'General': {'Print Verbosity':0, #set to 1 if you struggle to understand the output
                'Secant': {'Type': 'Limited-Memory BFGS', 'Maximum Storage': 10}},
    'Step': {'Type': 'Augmented Lagrangian',
             'Augmented Lagrangian': {'Subproblem Step Type': 'Trust Region',
                                       'Print Intermediate Optimization History': True,
                                       'Subproblem Iteration Limit': 3}},
    'Status Test': {'Gradient Tolerance': 1e-18,
                    #'Step Tolerance': 1e-14,
                    'Constraint Tolerance': 1e-12,
                    'Iteration Limit': 3}
                    }
    #print('set params', flush=True)
    params = ROL.ParameterList(params_dict, "Parameters")
    #print('set problem', flush=True)
    problem = ROL.OptimizationProblem(J, q, econ=econ, emul=emul)
    #print('set solver', flush=True)
    solver = ROL.OptimizationSolver(problem, params)
    #import ipdb
    #ipdb.set_trace()
    solver.solve()
