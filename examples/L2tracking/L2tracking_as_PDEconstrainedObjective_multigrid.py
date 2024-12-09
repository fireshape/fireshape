import firedrake as fd
import fireshape as fs
import ROL


class L2tracking(fs.PDEconstrainedObjective):
    """A Poisson BVP with hom DirBC as PDE constraint."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.mesh = self.Q.mesh_m

        # Setup problem
        self.V = fd.FunctionSpace(self.mesh_m, "CG", 1)

        # Preallocate solution variables for state and adjoint equations
        self.solution = fd.Function(self.V, name="State")

        # Weak form of Poisson problem
        u = self.solution
        v = fd.TestFunction(self.V)
        self.f = fd.Constant(4.)
        self.F = (fd.inner(fd.grad(u), fd.grad(v)) - self.f * v) * fd.dx
        self.bcs = fd.DirichletBC(self.V, 0., "on_boundary")

        # PDE-solver parameters
        self.params = {"mat_type": "aij",
                       "ksp_type": "cg",
                       # "ksp_view": None,
                       # "ksp_monitor": None,
                       "ksp_rtol": 1.0e-14,
                       "pc_type": "mg",
                       "mg_coarse_pc_type": "cholesky",
                       "mg_levels": {"ksp_max_it": 1,
                                     "ksp_type": "chebyshev",
                                     "pc_type": "jacobi"}
                       }

        stateproblem = fd.NonlinearVariationalProblem(
            self.F, self.solution, bcs=self.bcs)
        self.solver = fd.NonlinearVariationalSolver(
            stateproblem, solver_parameters=self.params)

        # target function, exact soln is disc of radius 0.6 centered at
        # (0.5,0.5)
        (x, y) = fd.SpatialCoordinate(self.mesh_m)
        self.u_target = 0.36 - (x-0.5)*(x-0.5) - (y-0.5)*(y-0.5)

    def objective_value(self):
        """Evaluate misfit functional. Signature imposed by ROL."""
        self.solver.solve()
        u = self.solution
        return fd.assemble((u - self.u_target)**2 * fd.dx)


if __name__ == '__main__':
    # setup problem
    mesh = fd.UnitSquareMesh(5, 5, diagonal="crossed")
    nref = 2
    Q = fs.FeMultiGridControlSpace(mesh, refinements=nref)
    inner = fs.H1InnerProduct(Q)
    q = fs.ControlVector(Q, inner)

    # create PDEconstrained objective functional
    names = ["domain"+str(ii)+".pvd" for ii in range(nref+1)]
    out = [fd.VTKFile(name) for name in names]

    def cb(*args):
        for ii, D in enumerate(Q.mh_mapped):
            out[ii].write(D.coordinates)
            print("Vol(", ii, ") = ", fd.assemble(fd.Constant(1.)*fd.dx(D)))
    J = L2tracking(Q, cb=cb)

    # ROL parameters
    params_dict = {
        'General': {
            'Secant': {
                'Type': 'Limited-Memory BFGS',
                'Maximum Storage': 10
            }
        },
        'Step': {
            'Type': 'Line Search',
            'Line Search': {
                'Descent Method': {
                    'Type': 'Quasi-Newton Step'
                }
            },
        },
        'Status Test': {
            'Gradient Tolerance': 1e-4,
            'Step Tolerance': 1e-5,
            'Iteration Limit': 15
        }
    }

    # assemble and solve ROL optimization problem
    params = ROL.ParameterList(params_dict, "Parameters")
    problem = ROL.OptimizationProblem(J, q)
    solver = ROL.OptimizationSolver(problem, params)
    solver.solve()

    vol = [fd.assemble(fd.Constant(1)*fd.dx(D)) for D in Q.mh_mapped]
    import numpy as np
    assert np.allclose(vol, vol[0])
