from firedrake import *
from fireshape import *
import fireshape.zoo as fsz
import ROL


class Compliance(PDEconstrainedObjective):
    """Minimize compliance of cantilever."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        V = VectorFunctionSpace(self.Q.mesh_m, "CG", 1)

        # no displacement at the back
        bcs = DirichletBC(V, Constant(as_vector([0, 0, 0])), 1)

        # gravity and elasticity parameters
        rho = Constant(0.01)
        g = Constant(1)
        f = as_vector([0, -rho*g, 0])  # gravity load
        p = 2*f  # extra load at tip (bdflag 2)
        mu = Constant(1)
        lambda_ = Constant(0.25)

        # weak formulation
        Id = Identity(mesh.geometric_dimension())  # 2x2 Identity tensor
        self.e = lambda u: 0.5*(grad(u) + grad(u).T)
        self.s = lambda u: lambda_*div(u)*Id + 2*mu*self.e(u)
        v = TestFunction(V)
        u = Function(V)
        self.u = u
        F = ((inner(self.s(u), self.e(v)) - dot(f, v))*dx - dot(p, v)*ds(2))
        prb = NonlinearVariationalProblem(F, self.u, bcs=bcs)
        self.solver = NonlinearVariationalSolver(prb)

        # define self.cb, which is always called after a domain update
        File = VTKFile("domain.pvd")
        self.cb = lambda: File.write(self.Q.mesh_m.coordinates)

    def visualize_displacement(self, name=None):
        """
        Solve linear elasticity equations. If input name is passed, store
        displacement in corresponding file.
        """
        self.solver.solve()
        VTKFile(name + "_displacement.pvd").write(self.u)

    def objective_value(self):
        """Return the value of the objective function."""
        self.solver.solve()
        eps_u = self.e(self.u)
        sigma_u = self.s(self.u)
        return assemble(inner(eps_u, sigma_u)*dx)


if __name__ == "__main__":

    # setup problem
    mesh = Mesh("mesh_cantilever.msh")
    Q = FeControlSpace(mesh)
    # do not modify shape of back and tip
    innerprod = H1InnerProduct(Q, fixed_bids=[1, 2])
    q = ControlVector(Q, innerprod)
    J = Compliance(Q) + fsz.MoYoSpectralConstraint(10, Constant(0.5), Q)

    # Set up volume constraint
    vol = fsz.VolumeFunctional(Q)
    vol0 = vol.value(q, None)
    C = EqualityConstraint([vol], target_value=[vol0])
    M = ROL.StdVector(1)

    # ROL parameters
    pd = {'General': {'Secant': {'Type': 'Limited-Memory BFGS'}},
          'Step': {'Type': 'Augmented Lagrangian',
                   'Augmented Lagrangian':
                     {'Subproblem Step Type': 'Trust Region',
                      'Subproblem Iteration Limit': 5}},
          'Status Test': {'Gradient Tolerance': 1e-3,
                          'Step Tolerance': 1e-3,
                          'Constraint Tolerance': 1e-3,
                          'Iteration Limit': 10}}
    params = ROL.ParameterList(pd, "Parameters")
    problem = ROL.OptimizationProblem(J, q, econ=C, emul=M)
    solver = ROL.OptimizationSolver(problem, params)

    # visualize initial displacement
    J.a.visualize_displacement("initial")
    print("Optimize domain")
    solver.solve()
    J.a.visualize_displacement("final")
