import firedrake as fd

class PdeConstraint(object):

    """
    Base class for pde constraint. Needs to implement the
    folllowing properties:

        F
        V
        bcs
        nsp
        params
        solution
        solution_adj

    """

    def __init__(self):
        self.num_solves = 0
        self.num_adjoint_solves = 0

    def solve(self):
        self.num_solves += 1

    def solve_adjoint(self, J):
        self.num_adjoint_solves += 1
        # Check if F is linear i.e. using TrialFunction
        if len(self.F.arguments()) != 2:
            bil_form = fd.derivative(self.F, self.solution,
                                     fd.TrialFunction(self.V))
        else:
            bil_form = fd.lhs(self.F)
        a = fd.adjoint(bil_form)
        rhs = -fd.derivative(J, self.solution, fd.TestFunction(self.V))
        fd.solve(a == rhs, self.solution_adj, bcs=fd.homogenize(self.bcs),
                 nullspace=self.nsp, transpose_nullspace=self.nsp,
                 solver_parameters=self.params)
        return self.solution_adj

    def derivative_form(self, v):
        raise NotImplementedError
