import pytest
import firedrake as fd
import fireshape as fs
import ROL
import numpy as np


@pytest.mark.parametrize("dim", [2, 3])
@pytest.mark.parametrize("inner_t", [fs.H1InnerProduct,
                                     fs.ElasticityInnerProduct,
                                     fs.LaplaceInnerProduct])
@pytest.mark.parametrize("use_extension", ["wo_ext", "w_ext",
                                           "w_ext_fixed_fim"])
def test_periodic(dim, inner_t, use_extension, pytestconfig):
    verbose = pytestconfig.getoption("verbose")
    """ Test template for PeriodicControlSpace."""

    if dim == 2:
        mesh = fd.PeriodicUnitSquareMesh(30, 30)
    elif dim == 3:
        mesh = fd.PeriodicUnitCubeMesh(20, 20, 20)
    else:
        raise NotImplementedError

    Q = fs.PeriodicControlSpace(mesh)

    inner = inner_t(Q)

    # levelset test case
    V = fd.FunctionSpace(Q.mesh_m, "DG", 0)
    sigma = fd.Function(V)
    if dim == 2:
        x, y = fd.SpatialCoordinate(Q.mesh_m)
        g = fd.sin(y*np.pi)  # truncate at bdry
        f = fd.cos(2*np.pi*x)*g
        perturbation = 0.05*fd.sin(x*np.pi)*g**2
        sigma.interpolate(g*fd.cos(2*np.pi*x*(1+perturbation)))
    elif dim == 3:
        x, y, z = fd.SpatialCoordinate(Q.mesh_m)
        g = fd.sin(y*np.pi)*fd.sin(z*np.pi)  # truncate at bdry
        f = fd.cos(2*np.pi*x)*g
        perturbation = 0.05*fd.sin(x*np.pi)*g**2
        sigma.interpolate(g*fd.cos(2*np.pi*x*(1+perturbation)))
    else:
        raise NotImplementedError

    class LevelsetFct(fs.ShapeObjective):
        def __init__(self, sigma, f, *args, **kwargs):
            super().__init__(*args, **kwargs)

            self.sigma = sigma  # initial
            self.f = f          # target
            Vdet = fd.FunctionSpace(Q.mesh_r, "DG", 0)
            self.detDT = fd.Function(Vdet)

        def value_form(self):
            # volume integral
            self.detDT.interpolate(fd.det(fd.grad(self.Q.T)))
            if min(self.detDT.vector()) > 0.05:
                integrand = (self.sigma - self.f)**2
            else:
                integrand = np.nan*(self.sigma - self.f)**2
            return integrand * fd.dx(metadata={"quadrature_degree": 1})

    # if running with -v or --verbose, then export the shapes
    if verbose:
        out = fd.File("sigma.pvd")

        def cb(*args):
            out.write(sigma)
    else:
        cb = None
    J = LevelsetFct(sigma, f, Q, cb=cb)

    if use_extension == "w_ext":
        ext = fs.ElasticityExtension(Q.V_r)
    if use_extension == "w_ext_fixed_dim":
        ext = fs.ElasticityExtension(Q.V_r, fixed_dims=[0])
    else:
        ext = None

    q = fd.ControlVector(Q, inner, boundary_extension=ext)

    """
    move mesh a bit to check that we are not doing the
    taylor test in T=id
    """
    g = q.clone()
    J.gradient(g, q, None)
    q.plus(g)
    J.update(q, None, 1)

    """ Start taylor test """
    J.gradient(g, q, None)
    res = J.checkGradient(q, g, 5, 1)
    errors = [l[-1] for l in res]
    assert (errors[-1] < 0.11 * errors[-2])
    q.scale(0)
    """ End taylor test """

    # ROL parameters
    grad_tol = 1e-4
    params_dict = {'Step': {'Type': 'Trust Region'},
                   'General': {'Secant': {'Type': 'Limited-Memory BFGS',
                                          'Maximum Storage': 25}},
                   'Status Test': {'Gradient Tolerance': grad_tol,
                                   'Step Tolerance': 1e-10,
                                   'Iteration Limit': 40}}

    # assemble and solve ROL optimization problem
    params = ROL.ParameterList(params_dict, "Parameters")
    problem = ROL.OptimizationProblem(J, q)
    solver = ROL.OptimizationSolver(problem, params)
    solver.solve()

    # verify that the norm of the gradient at optimum is small enough
    state = solver.getAlgorithmState()
    assert (state.gnorm < grad_tol)


if __name__ == '__main__':
    pytest.main()
