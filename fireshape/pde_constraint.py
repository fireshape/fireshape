import firedrake as fd


class PdeConstraint(object):
    """
    Base class for PdeConstrain.

    An instance of this class need to declare the following variables:
        mesh_m: the physical mesh.
        F: UFL form the represent state equation as zero search.
        V: trial and test space of state equation.
        bcs: boundary conditions of state/adjoint equation.
        nsp: null-space of the state operator.
        params: parameters used by fd.solve to solve F(x) = 0.
        solution: container for solution to state equatim(on.
        testfunction: the testfunction used in the weak formulation.
        solution_adj: containter for solution to adjoint equation.
    """

    def __init__(self):
        """Set counters of state/adjoint solves to 0."""
        self.num_solves = 0
        self.num_adjoint_solves = 0
        self.solver_adj = None
        self.solution = None
        self.solution_adj = None

    def solve(self):
        """Abstract method that solves state equation."""
        self.num_solves += 1
        self.solver.solve()
        return self.solution

    def solve_adjoint(self, J):
        """Abstract method that solves adjoint (and state) equation."""
        if self.solution is None:
            self.solution = self.solver._problem.u
        if self.solution_adj is None:
            self.solution_adj = self.solution.copy(deepcopy=True)

        if self.solver_adj is None:
            problem = self.solver._problem
            solver = self.solver
            ctx = solver._ctx
            bcs_adj = fd.homogenize(problem.bcs)

            F = self.solver._problem.F
            if len(self.F.arguments()) == 1:
                test = F.arguments()[0]
                F = fd.replace(F, {test: self.solution_adj})
                F = fd.derivative(F, self.solution)
                # F += fd.derivative(J, self.solution)
            else:
                raise NotImplementedError
            #     F += fd.derivative(J, self.solution)

            problem_adj = fd.NonlinearVariationalProblem(
                F, self.solution_adj,
                bcs=bcs_adj,# J=problem.J,
                #Jp=problem.Jp,
                form_compiler_parameters=problem.form_compiler_parameters)
            self.solver_adj = fd.NonlinearVariationalSolver(
                problem_adj,
                nullspace=ctx._nullspace_T,
                transpose_nullspace=ctx._nullspace,
                near_nullspace=ctx._near_nullspace,
                solver_parameters=self.params,
                appctx=ctx.appctx,
                # options_prefix=solver.options_prefix + "_adjoint",
                pre_jacobian_callback=ctx._pre_jacobian_callback,
                pre_function_callback=ctx._pre_function_callback)



        F = self.solver._problem.F

        if len(self.F.arguments()) == 1:
            # test = F.arguments()[0]
            # F = fd.replace(F, {test: self.solution_adj})
            # F = fd.derivative(F, self.solution)
            F += fd.derivative(J, self.solution)
        else:
            raise NotImplementedError(
                "Haven't thought about the case when F contains a TrialFunction."
            )

        self.solver_adj._problem.F = F
        # self.solver_adj._problem.J = fd.derivative(self.solver_adj._problem.J, self.solution_adj)
        self.solution_adj.assign(0)
        self.solver_adj.solve()
        return self.solution_adj

    def derivative_form(self, v):
        """Shape directional derivative of self.F."""
        X = fd.SpatialCoordinate(self.mesh_m)
        F = self.solver._problem.F
        test = F.arguments()[0]
        L = fd.replace(F, {test: self.solution_adj})
        return fd.derivative(L, X, v)
