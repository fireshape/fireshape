import firedrake as fd
from fireshape import ShapeObjective
from PDEconstraint_pipe import NavierStokesSolver
import numpy as np


class PipeObjective(ShapeObjective):
    """L2 tracking functional for Poisson problem."""

    def __init__(self, pde_solver: NavierStokesSolver, *args,  **kwargs):
        super().__init__(*args, **kwargs)
        self.pde_solver = pde_solver

    def value_form(self):
        """Evaluate misfit functional."""
        nu = self.pde_solver.viscosity

        if self.pde_solver.failed_to_solve: #is this a good solution
            self.pde_solver.failed_to_solve = False #I don't like resetting it here, but otherwise it doesn't work :(
#use self.pde_solver.failed_to_solve = True to replicate the error
            return np.nan * fd.dx(self.pde_solver.mesh_m)#is it a problem when we differentiate?
#the following example shows that the if J = nan for one radius,
#rol may not solve the state eqn again => self.pde_solver.failed_to_solve remain True
#and we get NaNs when trying to computed Riesz representatives. How to avoid this?
#(firedrake) (ap/pipe_new *)$ python3 main_pipe.py 
#
# Augmented Lagrangian Solver
#Subproblem Solver: Trust Region
#  iter  fval           cnorm          gLnorm         snorm          penalty   feasTol   optTol    #fval   #grad   #cval   subIter 
#  0     4.390787e-01   0.000000e+00   2.032227e-01                  1.00e+01  1.26e-01  1.00e-02  
#
#Dogleg Trust-Region Solver with Limited-Memory BFGS Hessian Approximation
#  iter  value          gnorm          snorm          delta          #fval     #grad     tr_flag   
#  0     4.390787e-01   2.032227e-01                  2.032227e-01   
#  1     3.883671e-01   1.317248e-01   2.032227e-01   5.080567e-01   3         2         0         
#  2     3.597674e-01   4.381117e-01   3.930074e-01   1.270142e+00   4         3         0         
#  3     3.597674e-01   4.381117e-01   1.250172e-01   3.125430e-02   5         3         2         
#  4     3.517573e-01   1.746985e-01   3.125430e-02   3.125430e-02   6         4         0         
#  5     3.511796e-01   9.376058e-02   3.125430e-02   3.125430e-02   7         5         0         
#Optimization Terminated with Status: Iteration Limit Exceeded
#  1     3.462729e-01   3.240761e-01   3.067325e-01   1.563136e+00   1.00e+01  1.00e-01  2.03e-04  10      7       13      5       
#
#Dogleg Trust-Region Solver with Limited-Memory BFGS Hessian Approximation
#  iter  value          gnorm          snorm          delta          #fval     #grad     tr_flag   
#  0     4.934724e-01   1.900944e+00                  1.900944e+00   
#  1     4.934724e-01   1.900944e+00   1.900944e+00   3.456262e-01   3         1         5         
#  2     4.934724e-01   1.900944e+00   3.456262e-01   3.764246e-02   4         1         5         
#  3     4.934724e-01   1.900944e+00   3.764246e-02   3.798090e-03   5         1         5         
#  4     4.934724e-01   1.900944e+00   3.798090e-03   3.801508e-04   6         1         5         
#  5     4.934724e-01   1.900944e+00   3.801508e-04   3.801850e-05   7         1         5         
#Optimization Terminated with Status: Iteration Limit Exceeded
#Traceback (most recent call last):
#  File "main_pipe.py", line 73, in <module>
#    solver.solve()
#  File "/Users/alberto/Documents/FIREDRAKE/fireshape/fireshape/objective.py", line 63, in gradient
#    g.apply_riesz_map()
#  File "/Users/alberto/Documents/FIREDRAKE/fireshape/fireshape/control.py", line 628, in apply_riesz_map
#    self.inner_product.riesz_map(self, self)
#  File "/Users/alberto/Documents/FIREDRAKE/fireshape/fireshape/innerproduct.py", line 167, in riesz_map
#    self.ls.solve(out.fun, v.fun)
#  File "/Users/alberto/Documents/FIREDRAKE/firedrake/firedrake/src/firedrake/firedrake/linear_solver.py", line 160, in solve
#    raise ConvergenceError("LinearSolver failed to converge after %d iterations with reason: %s", self.ksp.getIterationNumber(), solving_utils.KSPReasons[r])
#firedrake.exceptions.ConvergenceError: ('LinearSolver failed to converge after %d iterations with reason: %s', 7, 'DIVERGED_NANORINF')
        else:
            z = self.pde_solver.solution
            u, p = fd.split(z)
            return nu * fd.inner(fd.grad(u), fd.grad(u)) * fd.dx
