import ROL

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

    def plus(self, v):
        #print('adding', flush=True)
        self.x += v.x

    def scale(self, alpha):
        #print('scaling', flush=True)
        self.x *= alpha

    def clone(self):
        #print('cloning', flush=True)
        return ControlVector()

    def dot(self, v):
        #print('dotting', flush=True)
        return self.x * v.x

    def norm(self):
        #print('norming', flush=True)
        return abs(self.x)

    def axpy(self, alpha, y):
        self.x = self.x*alpha + y.x

    def set(self, v):
        #print('setting', flush=True)
        self.x = v.x

class Objective(ROL.Objective):
    def __init__(self, cb=None):
        super().__init__()

    def value(self, x, tol):
        return (x.x-1)**2

    def gradient(self, g, x, tol):
        g.x = 2.*x.x

if __name__== "__main__":
    #print('call ControlVector()', flush=True)
    q = ControlVector()
    #print('call Objective()', flush=True)
    J = Objective()
    params_dict = {
    'General': {'Print Verbosity':1, #set to 1 if you struggle to understand the output
                'Secant': {'Type': 'Limited-Memory BFGS', 'Maximum Storage': 10}},
    #'Step': {'Type': 'Augmented Lagrangian',
    #         'Augmented Lagrangian': {'Subproblem Step Type': 'Trust Region',
    #                                   'Print Intermediate Optimization History': True,
    #                                   'Subproblem Iteration Limit': 5}},
    'Status Test': {'Gradient Tolerance': 1e-2,
                    'Step Tolerance': 1e-2,
                    'Constraint Tolerance': 1e-1,
                    'Iteration Limit': 2}
                    }
    #print('set params', flush=True)
    params = ROL.ParameterList(params_dict, "Parameters")
    #print('set problem', flush=True)
    problem = ROL.OptimizationProblem(J, q)
    #print('set solver', flush=True)
    solver = ROL.OptimizationSolver(problem, params)
    #import ipdb
    #ipdb.set_trace()
    solver.solve()
