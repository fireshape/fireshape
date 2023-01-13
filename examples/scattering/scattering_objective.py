import firedrake as fd
from fireshape import PDEconstrainedObjective
import numpy as np
from scattering_PDEconstraint import PMLSolver
from utils import generate_mesh, plot_mesh, plot_field, plot_vector_field,\
                  plot_far_field


class FarFieldObjective(PDEconstrainedObjective):
    """Misfit of the far field pattern."""

    def __init__(self, pde_solver: PMLSolver, R0, R1, layer, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.e = pde_solver
        self.k = pde_solver.k
        self.dirs = pde_solver.dirs
        self.solutions = pde_solver.solutions
        self.R0 = R0
        self.R1 = R1
        self.layer = layer

        # Cut-off function for far field evaluation
        V = pde_solver.V
        mesh = V.mesh()
        y = fd.SpatialCoordinate(mesh)
        r = fd.sqrt(fd.dot(y, y))
        psi = (1 - fd.cos((r - R0) / (R1 - R0) * fd.pi)) / 2
        self.grad_psi = fd.Function(V, name="grad_psi")
        self.grad_psi.interpolate(fd.grad(psi))
        elem = V.ufl_element().sub_elements()[0]
        self.laplace_psi = fd.Function(fd.FunctionSpace(mesh, elem),
                                       name="laplace_psi")
        self.laplace_psi.interpolate(fd.div(fd.grad(psi)))

        # Product of complex numbers
        def dot(x, y):
            x_re, x_im = x
            y_re, y_im = y
            res_re = x_re * y_re - x_im * y_im
            res_im = x_re * y_im + x_im * y_re
            return res_re, res_im

        # Form of far field
        k = fd.Constant(self.k)
        x_hat = fd.Constant((0, 0))
        coeff = 1 / np.sqrt(8 * np.pi * self.k)
        u = fd.Function(V)

        phi = fd.pi / 4 - k * fd.dot(x_hat, y)
        f = (fd.cos(phi), fd.sin(phi))
        g = dot(u, (self.laplace_psi, -2 * k * fd.dot(x_hat, self.grad_psi)))
        h = dot(f, g)
        self.integrand = (coeff * h[0], coeff * h[1])

        # Save placeholders
        self.dx = fd.dx(2, metadata={"quadrature_degree": 4})
        self.x_hat = x_hat
        self.y = y
        self.u = u

        # Points on S^1
        n = self.n = 50
        self.weight = 2 * np.pi / self.n / pde_solver.n_wave
        theta = 2 * np.pi / n * np.arange(n)
        self.xs = []
        for i in range(n):
            t = theta[i]
            self.xs.append(fd.Constant((np.cos(t), np.sin(t))))

        # Target far field pattern
        self.g = self.target_far_field()

    def target_far_field(self):
        """
        Evaluate far field pattern on the target mesh.
        """
        use_cached_mesh = False
        a0 = b0 = 2.0
        a1 = b1 = 2.25
        R0 = 1.5
        R1 = 1.9
        layer = [(a0, a1), (b0, b1)]

        if use_cached_mesh:
            mesh = fd.Mesh("target.msh", name="target")
        else:
            obstacle = {
                "shape": "circle",
                "center": (0.1, 0.),
                "scale": 0.4
            }
            mesh = generate_mesh(obstacle, layer, R0, R1, 3, name="target")
        plot_mesh(mesh, self.Q.bbox, "target")

        solver = PMLSolver(mesh, self.k, self.dirs, a1, b1)
        solver.solve()
        u_target = solver.solutions
        plot_field(u_target[0], layer, "u_target")
        g = self.far_field(u_target, R0, R1)
        return g

    def far_field(self, near_fields, R0, R1):
        """
        Evaluate far field of the scattered wave.
        """
        V = near_fields[0].function_space()
        mesh = V.mesh()

        # Integrand of far field pattern
        if mesh == self.grad_psi.function_space().mesh():
            expr = self.integrand
        else:
            # Replace cut-off function for target mesh
            y = fd.SpatialCoordinate(mesh)
            r = fd.sqrt(fd.dot(y, y))
            psi = (1 - fd.cos((r - R0) / (R1 - R0) * fd.pi)) / 2
            grad_psi = fd.interpolate(fd.grad(psi), V)
            elem = V.ufl_element().sub_elements()[0]
            laplace_psi = fd.interpolate(fd.div(fd.grad(psi)),
                                         fd.FunctionSpace(mesh, elem))
            replacement = {self.y: y,
                           self.grad_psi: grad_psi,
                           self.laplace_psi: laplace_psi}
            expr = (fd.replace(self.integrand[0], replacement),
                    fd.replace(self.integrand[1], replacement))

        far_fields = []
        for u in near_fields:
            form = (fd.replace(expr[0], {self.u: u}) * self.dx,
                    fd.replace(expr[1], {self.u: u}) * self.dx)
            u_inf = []
            for i in range(self.n):
                self.x_hat.assign(self.xs[i])
                u_inf.append((fd.assemble(form[0]), fd.assemble(form[0])))
            far_fields.append(u_inf)
        return far_fields

    def objective_value(self):
        """
        Evaluate reduced objective. Method introduced to
        bypass ROL signature in self.value.
        """
        res = 0
        for i in range(self.e.n_wave):
            u_inf = self.u_inf[i]
            g = self.g[i]
            for j in range(self.n):
                res += (u_inf[j][0] - g[j][0])**2 + (u_inf[j][1] - g[j][1])**2
        return res * self.weight

    def solvePDE(self):
        """Solve the PDE constraint."""
        self.e.solve()
        self.u_inf = self.far_field(self.solutions, self.R0, self.R1)

    def derivative(self, out):
        """
        Get the derivative from pyadjoint.
        """
        deriv = self.Jred.derivative()
        out.from_first_derivative(deriv)

        plot_field(self.solutions[0], self.layer, "u")
        plot_far_field(self.u_inf[0], self.g[0], "u_inf")
        plot_vector_field(deriv, self.layer, self.Q.bbox, "deriv")

    def gradient(self, g, x, tol):
        super().gradient(g, x, tol)

        T = fd.Function(self.Q.T)
        g.to_coordinatefield(T)
        plot_vector_field(T, self.layer, self.Q.bbox, "grad")
