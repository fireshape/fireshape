import firedrake as fd
from fireshape import PdeConstraint


class PMLSolver(PdeConstraint):
    """Acoustic scattering problem from a sound-soft obstacle solved by PML."""
    def __init__(self, mesh_m, k, d, a1, b1):
        super().__init__()

        # Setup problem
        V = fd.VectorFunctionSpace(mesh_m, "CG", 1)
        X = fd.SpatialCoordinate(mesh_m)
        u = fd.Function(V, name="State")
        v = fd.TestFunction(V)
        self.k = k
        self.d = d
        k = fd.Constant(k)
        d = fd.as_vector(d)

        # Subdomains
        dx_F = fd.dx(1) + fd.dx(2)  # physical domain
        dx_A_x = fd.dx(3)  # vertical layers
        dx_A_y = fd.dx(4)  # horizontal layers
        dx_A_xy = fd.dx(5)  # corner layers

        # Coefficients
        sigma_x = 1 / k / (a1 - abs(X[0]))
        sigma_y = 1 / k / (b1 - abs(X[1]))
        # c1 = gamma_y / gamma_x
        c1_x = (1 / (1 + sigma_x**2), -sigma_x / (1 + sigma_x**2))
        c1_y = (1, sigma_y)
        c1_xy = ((1 + sigma_x * sigma_y) / (1 + sigma_x**2),
                 (sigma_y - sigma_x) / (1 + sigma_x**2))
        # c2 = gamma_x / gamma_y
        c2_x = (1, sigma_x)
        c2_y = (1 / (1 + sigma_y**2), -sigma_y / (1 + sigma_y**2))
        c2_xy = ((1 + sigma_x * sigma_y) / (1 + sigma_y**2),
                 (sigma_x - sigma_y) / (1 + sigma_y**2))
        # c3 = gamma_x * gamma_y
        c3_x = (1, sigma_x)
        c3_y = (1, sigma_y)
        c3_xy = (1 - sigma_x * sigma_y, sigma_x + sigma_y)

        # Evaluate sesquilinear form with coefficient
        def inner(x, y, c=None):
            x_re, x_im = x
            y_re, y_im = y
            y_im = -y_im
            if c:
                c_re, c_im = c
                res_re = c_re * x_re * y_re - c_re * x_im * y_im\
                    - c_im * x_re * y_im - c_im * x_im * y_re
                res_im = c_re * x_re * y_im + c_re * x_im * y_re\
                    + c_im * x_re * y_re - c_im * x_im * y_im
            else:
                res_re = x_re * y_re - x_im * y_im
                res_im = x_re * y_im + x_im * y_re
            return res_re + res_im

        # Weak form
        ux, uy = u.dx(0), u.dx(1)
        vx, vy = v.dx(0), v.dx(1)
        self.F = (inner(ux, vx) + inner(uy, vy) - k**2 * inner(u, v)) * dx_F\
            + (inner(ux, vx, c=c1_x) + inner(uy, vy, c=c2_x)
                - k**2 * inner(u, v, c=c3_x)) * dx_A_x\
            + (inner(ux, vx, c=c1_y) + inner(uy, vy, c=c2_y)
                - k**2 * inner(u, v, c=c3_y)) * dx_A_y\
            + (inner(ux, vx, c=c1_xy) + inner(uy, vy, c=c2_xy)
                - k**2 * inner(u, v, c=c3_xy)) * dx_A_xy
        # boundary condition
        kdx = k * fd.dot(d, X)
        u_inc = fd.as_vector((fd.cos(kdx), fd.sin(kdx)))
        self.bcs = [fd.DirichletBC(V, -u_inc, 1),
                    fd.DirichletBC(V, (0., 0.), 5)]

        # PDE-solver parameters
        self.params = {
            "ksp_type": "cg",
            "ksp_rtol": 1e-11,
            "ksp_atol": 1e-11,
            "ksp_stol": 1e-15,
        }

        self.solution = u

    def solve(self):
        super().solve()
        fd.solve(self.F == 0, self.solution, bcs=self.bcs,
                 solver_parameters=self.params)
