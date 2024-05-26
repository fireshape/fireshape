import firedrake as fd
import fireshape as fs
import petsc4py.PETSc as PETSc

class L2tracking(fs.PDEconstrainedObjective):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # target solution
        x, y = fd.SpatialCoordinate(self.Q.mesh_m)
        self.u_target = 0.36 - (x-0.5)*(x-0.5) - (y-0.5)*(y-0.5)

        # Poisson problem with homogeneous DirichletBC
        V = fd.FunctionSpace(self.Q.mesh_m, "CG", 1)
        u = fd.Function(V)
        v = fd.TestFunction(V)
        f = fd.Constant(4.)
        F = (fd.dot(fd.grad(u), fd.grad(v)) - f * v) * fd.dx
        self.bcs = fd.DirichletBC(V, 0, "on_boundary")
        self.u = u  # make it accessible to self.solvePDE
        self.u_old = fd.Function(V)  # solution at previous iteration
        problem = fd.NonlinearVariationalProblem(F, u, bcs=self.bcs)
        self.solver = fd.NonlinearVariationalSolver(problem)
        self.solver.solve()

        # define self.cb to store function and mesh at each iteration
        out = fd.VTKFile("soln.pvd")
        self.cb = lambda: out.write(self.u)

        # function space and variable to assess mesh quality
        Vdet = fd.FunctionSpace(self.Q.mesh_r, "DG", 0)
        self.detDT = fd.Function(Vdet)

    def solvePDE(self):
        """Solve the heat equation and evaluate the objective function."""
        try:
            self.solver.solve()
            self.PDEsolved = True
            self.u_old.assign(self.u)
        except fd.ConvergenceError:
            self.PDEsolved = False
            self.u.assign(self.u_old)

    def objective_value(self):
        """Return the value of the objective function."""
        self.detDT.interpolate(fd.det(fd.grad(self.Q.T)))
        mesh_is_fine = min(self.detDT.vector()) > 0.01
        if mesh_is_fine and self.PDEsolved:
            return fd.assemble((self.u - self.u_target)**2 * fd.dx)
        else:
            from pyadjoint.adjfloat import AdjFloat
            import numpy as np
            return AdjFloat(np.NAN)

if __name__=="__main__":
    # setup problem
    mesh = fd.UnitSquareMesh(100, 100)
    Q = fs.FeControlSpace(mesh)
    Q.assign_inner_product(fs.H1InnerProduct(Q))
    J = L2tracking(Q)

    # PETSc.TAO solver using the limited-memory
    # variable-metric method. Call using
    # python L2tracking.py -tao_monitor
    # to print updates in the terminal
    solver = PETSc.TAO().create()
    solver.setType("lmvm")
    solver.setFromOptions()
    solver.setSolution(Q.get_PETSc_zero_vec())
    solver.setObjectiveGradient(J.objectiveGradient, None)
    solver.setTolerances(gatol=1.0e-4, grtol=1.0e-4)
    solver.solve()
