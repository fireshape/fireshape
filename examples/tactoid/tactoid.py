from firedrake import *
#from fireshape import *

#class NematicFrankOseenSolver(PdeConstraint):
class NematicFrankOseenSolver(object):
    def __init__(self, mesh):
        self.mesh = mesh

        V = VectorFunctionSpace(mesh, "CG", 2, dim=2)
        Q = FunctionSpace(mesh, "CG", 1)
        Z = MixedFunctionSpace([V, Q])

        z = Function(Z)
        (n, l) = split(z)
        s = FacetNormal(mesh)         # normal vector
        t = as_vector([-s[1], s[0]])  # tangential vector

        K = Constant(1)
        W = Constant(5)

        J_nem = (
                  0.5 * K * inner(grad(n), grad(n))*dx
                + 0.5 * W * inner(dot(n, s), dot(n, s))*ds
                )
        L_nem = J_nem - inner(l, dot(n, n) - 1)*dx
        F = derivative(L_nem, z)

        sp = {"snes_type": "newtonls",
              "snes_monitor": None,
              "ksp_type": "preonly",
              "pc_type" : "lu",
              "pc_factor_mat_solver_type": "mumps",
              "mat_mumps_icntl_14": 200}

        problem = NonlinearVariationalProblem(F, z)
        solver = NonlinearVariationalSolver(problem, solver_parameters=sp)

        self.solver = solver
        self.solution = z

        z.sub(0).interpolate(as_vector([1, 0]))

    def solve(self):
        self.solver.solve()

if __name__ == "__main__":
    mesh = Mesh("disk.msh")
    e = NematicFrankOseenSolver(mesh)
    e.solve()
    pvd = File("output/solution.pvd")
    pvd.write(e.solution.split()[0])
