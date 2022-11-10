from firedrake import *
from fireshape import *
import fireshape.zoo as zoo

from numpy import nan
import ROL

class NematicObjective(PDEconstrainedObjective):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        mesh = self.Q.mesh_m
        self.failed_to_solve = False

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
        self.J_nem = J_nem

        sp = {"snes_type": "newtonls",
              "snes_linesearch_type": "l2",
              "snes_monitor": None,
              "snes_linesearch_monitor": None,
              "ksp_type": "preonly",
              "pc_type" : "lu",
              "pc_factor_mat_solver_type": "mumps",
              "mat_mumps_icntl_14": 200}

        problem = NonlinearVariationalProblem(F, z)
        solver = NonlinearVariationalSolver(problem, solver_parameters=sp)

        self.solver = solver

        (x, y) = SpatialCoordinate(mesh)
        theta = atan_2(y, x)
        z.sub(0).interpolate(as_vector([-sin(theta), cos(theta)]))

        self.solution = z
        self.solution_old = Function(z)

        self.pvd = File("output/solution.pvd")
        def callback():
            (n, l) = self.solution.split()
            n.rename("Director")
            self.pvd.write(n)
        self.cb = callback

    def solvePDE(self):
        self.failed_to_solve = False
        self.solution_old.assign(self.solution)
        try:
            self.solver.solve()
        except ConvergenceError:
            self.failed_to_solve = True
            self.solution.assign(self.solution_old)

    def objective_value(self):
        sigma = Constant(5.0*0.04)  # from morpho example
        mesh = self.Q.mesh_m

        if self.failed_to_solve:
            return assemble(nan*dx(mesh))
        else:
            J = (
                  self.J_nem  # nematic energy
                + sigma * ds(mesh)
                )

            J_out = assemble(J)

            area = assemble(Constant(1)*dx(mesh))
            area_penalty = 150 * (area - 2*pi)**2

            (x, y) = SpatialCoordinate(mesh)
            bary_x = assemble(x*dx)
            bary_y = assemble(y*dx)
            bary_penalty_x = 100 * (bary_x - 0)**2
            bary_penalty_y = 100 * (bary_y - 0)**2

            return J_out + area_penalty + bary_penalty_x + bary_penalty_y

if __name__ == "__main__":
    mesh = Mesh("disk.msh")
    Q = FeControlSpace(mesh)
    finner = LaplaceInnerProduct(Q)
    q = ControlVector(Q, finner)
    J = NematicObjective(Q)

    # Add regularisation to improve mesh quality
    Jq = zoo.MoYoSpectralConstraint(10, Constant(0.1), Q)
    J = J + Jq

    # Set up volume constraint
    #vol = zoo.VolumeFunctional(Q)
    #initial_vol = vol.value(q, None)
    #econ = EqualityConstraint([vol], target_value=[initial_vol])
    #emul = ROL.StdVector(1)

    # ROL parameters
    params_dict = {
        'Step': {
            'Type': 'Line Search',
            'Line Search': {
                'Descent Method': {
                    'Type': 'Quasi-Newton Step'
                }
            },
        },
        'General': {
            'Print Verbosity': 1,
            'Secant': {
                'Type': 'Limited-Memory BFGS',
                'Maximum Storage': 10
            }
        },
        'Status Test': {
            'Gradient Tolerance': 1e-4,
            'Step Tolerance': 1e-5,
            'Iteration Limit': 200
        }
    }
    params = ROL.ParameterList(params_dict, "Parameters")
    #problem = ROL.OptimizationProblem(J, q, econ=econ, emul=emul)
    problem = ROL.OptimizationProblem(J, q)
    solver = ROL.OptimizationSolver(problem, params)
    solver.solve()
