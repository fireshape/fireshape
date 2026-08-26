import firedrake as fd
from fireshape import PdeConstraint


class PMLSolver(PdeConstraint):
    """Acoustic scattering problem from a sound-soft obstacle solved by PML."""
    def __init__(self, mesh_m, k, dirs, a1, b1):
        super().__init__()
        self.k = k
        self.dirs = dirs
        self.n_wave = len(dirs)

        # Setup problem
        V = self.V = fd.VectorFunctionSpace(mesh_m, "CG", 1)
        X = fd.SpatialCoordinate(mesh_m)
        u = fd.TrialFunction(V)  # total field
        v = fd.TestFunction(V)

        self.d = fd.Constant((0, 0))  # placeholder for incident direction
        kdx = k * fd.dot(self.d, X)
        self.u_i = fd.as_vector((fd.cos(kdx), fd.sin(kdx)))  # incident field

        # Subdomains
        metadata = {"quadrature_degree": 4}
        dx_F = fd.dx(1, metadata=metadata) +\
            fd.dx(2, metadata=metadata)  # physical domain
        dx_A_x = fd.dx(3, metadata=metadata)  # vertical layers
        dx_A_y = fd.dx(4, metadata=metadata)  # horizontal layers
        dx_A_xy = fd.dx(5, metadata=metadata)  # corner layers

        # Coefficients
        k = fd.Constant(k)
        sigma_x = 1 / k / (fd.Constant(a1) - abs(X[0]))
        sigma_y = 1 / k / (fd.Constant(b1) - abs(X[1]))
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
        self.a = (inner(ux, vx) + inner(uy, vy) - k**2 * inner(u, v)) * dx_F\
            + (inner(ux, vx, c=c1_x) + inner(uy, vy, c=c2_x)
                - k**2 * inner(u, v, c=c3_x)) * dx_A_x\
            + (inner(ux, vx, c=c1_y) + inner(uy, vy, c=c2_y)
                - k**2 * inner(u, v, c=c3_y)) * dx_A_y\
            + (inner(ux, vx, c=c1_xy) + inner(uy, vy, c=c2_xy)
                - k**2 * inner(u, v, c=c3_xy)) * dx_A_xy
        self.L = fd.replace(self.a, {u: self.u_i})

        # Boundary conditions
        self.ds = []
        self.bcs = []
        bc_obs = fd.DirichletBC(self.V, (0., 0.), 1)  # sound-soft obstacle
        for d in dirs:
            d = fd.Constant(d)
            self.d.assign(d)
            self.ds.append(d)
            self.bcs.append([bc_obs, fd.DirichletBC(self.V, self.u_i, 5)])

        self.solutions = []
        for i in range(self.n_wave):
            self.solutions.append(fd.Function(V, name="State"+str(i)))

    def solve(self):
        super().solve()
        for i in range(self.n_wave):
            self.d.assign(self.ds[i])  # incident direction
            fd.solve(self.a == self.L, self.solutions[i], bcs=self.bcs[i])
            self.solutions[i].interpolate(self.solutions[i] - self.u_i)
