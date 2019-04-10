import pytest
import firedrake as fd
import fireshape as fs
from fireshape import ShapeObjective
from fireshape import PdeConstraint
import fireshape.zoo as fsz
import ROL


class PoissonSolver(PdeConstraint):
    """A Poisson BVP with hom DirBC as PDE constraint."""
    def __init__(self, mesh_m):
        super().__init__()
        self.mesh_m = mesh_m

        # Setup problem
        self.V = fd.FunctionSpace(self.mesh_m, "CG", 1)

        # Preallocate solution variables for state and adjoint equations
        self.solution = fd.Function(self.V, name="State")
        self.testfunction = fd.TestFunction(self.V)
        self.solution_adj = fd.Function(self.V, name="Adjoint")

        # Weak form of Poisson problem
        u = self.solution
        v = self.testfunction
        self.f = fd.Constant(4.)
        self.F = (fd.inner(fd.grad(u), fd.grad(v)) - self.f * v) * fd.dx
        self.bcs = fd.DirichletBC(self.V, 0., "on_boundary")

        # PDE-solver parameters
        self.nsp = None
        self.params = { 
                "ksp_type": "cg",
                "mat_type": "aij",
                "pc_type": "hypre",
                "pc_factor_mat_solver_package": "boomerang",
                "ksp_rtol": 1e-11,
                "ksp_atol": 1e-11,
                "ksp_stol": 1e-15,
                      }   

        stateproblem_ = fd.NonlinearVariationalProblem(self.F, self.solution, bcs=self.bcs)
        self.stateproblem = fd.NonlinearVariationalSolver(stateproblem_, solver_parameters=self.params)

class L2trackingObjective(ShapeObjective):
    """L2 tracking functional for Poisson problem."""
    def __init__(self, pde_solver: PoissonSolver, *args,  **kwargs):
        super().__init__(*args, **kwargs)
        self.pde_solver = pde_solver

        #target function, exact soln is disc of radius 0.6 centered at (0.5,0.5)
        (x, y) = fd.SpatialCoordinate(pde_solver.mesh_m)
        self.u_target = 0.36 - (x-0.5)*(x-0.5) - (y-0.5)*(y-0.5)

    def value_form(self):
        """Evaluate misfit functional."""
        u = self.pde_solver.solution
        return (u - self.u_target)**2 * fd.dx


def run_L2tracking_optimization(write_output=False):
    """ Test template for fsz.LevelsetFunctional."""

    #tool for developing new tests, allows storing shape iterates
    if write_output:
        out = fd.File("domain.pvd")

        def cb(*args):
            out.write(Q.mesh_m.coordinates)

        cb()
    else:
        cb = None

    #setup problem
    mesh = fd.UnitSquareMesh(30, 30)
    Q = fs.FeControlSpace(mesh)
    inner = fs.ElasticityInnerProduct(Q)
    q = fs.ControlVector(Q, inner)

    #setup PDE constraint
    mesh_m = Q.mesh_m
    e = PoissonSolver(mesh_m)

    #create PDEconstrained objective functional
    J_ = L2trackingObjective(e, Q, cb=cb)
    J = fs.ReducedObjective(J_, e)

    #ROL parameters
    params_dict = { 
        'General': {
            'Secant': {'Type': 'Limited-Memory BFGS',
                       'Maximum Storage': 10}},
        'Step': {
            'Type': 'Augmented Lagrangian',
            'Line Search': {'Descent Method': {
                'Type': 'Quasi-Newton Step'}
            },  
            'Augmented Lagrangian': {
                'Subproblem Step Type': 'Line Search',
                'Penalty Parameter Growth Factor': 2., 
                #'Print Intermediate Optimization History': False,
                'Subproblem Iteration Limit': 20
            }},
        'Status Test': {
            'Gradient Tolerance': 1e-4,
            'Step Tolerance': 1e-5,
            'Iteration Limit': 15}
    }

    # assemble and solve ROL optimization problem
    params = ROL.ParameterList(params_dict, "Parameters")
    problem = ROL.OptimizationProblem(J, q)
    solver = ROL.OptimizationSolver(problem, params)
    solver.solve()

    # verify that the norm of the gradient at optimum is small enough
    state = solver.getAlgorithmState()
    assert (state.gnorm < 1e-4)

def test_L2tracking(pytestconfig):
    verbose=False
    run_L2tracking_optimization(write_output=verbose)

if __name__ == '__main__':
    pytest.main()
