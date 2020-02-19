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
        self.x = 0.

    def plus(self, v):
        self.x += v.x

    def scale(self, alpha):
        self.x *= alpha

    def clone(self):
        return ControlVector()

    def dot(self, v):
        self.x *= v.x

    def axpy(self, alpha, y):
        self.x = self.x*alpha + y.x

    def set(self, v):
        self.x = v.x

class Objective(ROL.Objective):
    def __init__(self):
        super().__init__()

    def value(self, x, tol):
        return (x-1)**2

    def gradient(self, g, q, tol):
        g = 2.*q.x

if __name__== "__main__":

    q = ControlVector()
    J = Objective()
    params_dict = {
    'General': {'Print Verbosity':0, #set to 1 if you struggle to understand the output
                'Secant': {'Type': 'Limited-Memory BFGS', 'Maximum Storage': 10}},
    'Step': {'Type': 'Augmented Lagrangian',
             'Augmented Lagrangian': {'Subproblem Step Type': 'Trust Region',
                                       'Print Intermediate Optimization History': True,
                                       #'Subproblem Iteration Limit': 5}},
                                       'Subproblem Iteration Limit': 5}},
                                       #'Subproblem Iteration Limit': 10}}, #this fails with nans in computing grad
                                                                           #observation: a lot of subits lead to compressing
                                                                           #nodes in the middle of the pipe
    'Status Test': {'Gradient Tolerance': 1e-2,
                    'Step Tolerance': 1e-2,
                    'Constraint Tolerance': 1e-1,
                    'Iteration Limit': 2} #we can raise this to 100, nothing changes and it doesn't crash, it's good news, but finding appropri    ate stopping criteria is challenging
                    }
    params = ROL.ParameterList(params_dict, "Parameters")
    problem = ROL.OptimizationProblem(J, q)
    solver = ROL.OptimizationSolver(problem, params)
    solver.solve()
