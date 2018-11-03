import firedrake as fd


class PdeConstraint(object):
    """
    Base class for PdeConstrain.

    An instance of this class need to declare the following variables:
        F: UFL form the represent state equation as zero search.
        V: trial and test space of state equation.
        bcs: boundary conditions of state/adjoint equation.
        nsp: null-space of the state operator.
        params: parameters used by fd.solve to solve F(x) = 0.
        solution: container for solution to state equation.
        solution_adj: containter for solution to adjoint equation.
    """

    def __init__(self):
        """Set counters of state/adjoint solves to 0."""
        self.num_solves = 0
        self.num_adjoint_solves = 0
        self.form_compiler_params = None

    def solve(self):
        """Abstract method that solves state equation."""
        self.num_solves += 1

    def solve_adjoint(self, J):
        """Abstract method that solves adjoint (and state) equation."""
        self.num_solves += 1
        self.num_adjoint_solves += 1
        # Check if F is linear i.e. using TrialFunction
        if len(self.F.arguments()) == 2:
            bil_form = fd.lhs(self.F)
        else:
            bil_form = fd.derivative(self.F, self.solution,
                                     fd.TrialFunction(self.V))
        a = fd.adjoint(bil_form)
        rhs = -fd.derivative(J, self.solution, fd.TestFunction(self.V))
        A = fd.assemble(a, mat_type="aij", form_compiler_parameters=self.form_compiler_params)
        b = fd.assemble(rhs, form_compiler_parameters=self.form_compiler_params)
        fd.solve(A, self.solution_adj, b, bcs=fd.homogenize(self.bcs),
                 nullspace=self.nsp, transpose_nullspace=self.nsp,
                 solver_parameters=self.params, options_prefix=self.solver.options_prefix + "_adjoint")
        return self.solution_adj

    def derivative_form(self, v):
        raise NotImplementedError
