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
        self.u_s = pde_solver.solutions
        self.R0 = R0
        self.R1 = R1
        self.layer = layer

        # cut-off function for far field evaluation
        V = self.u_s[0].function_space()
        mesh = V.mesh()
        y = fd.SpatialCoordinate(mesh)
        r = fd.sqrt(fd.dot(y, y))
        psi = (1 - fd.cos((r - R0) / (R1 - R0) * fd.pi)) / 2
        self.grad_psi = fd.interpolate(fd.grad(psi), V)
        self.laplace_psi = fd.interpolate(fd.div(fd.grad(psi)),
                                          fd.FunctionSpace(mesh, "CG", 1))

        # points on S^1
        self.n = 50
        self.weight = 2 * np.pi / self.n / pde_solver.n_wave
        theta = np.linspace(0, 2 * np.pi, self.n).reshape(-1, 1)
        self.xs = np.hstack((np.cos(theta), np.sin(theta)))

        # target far field pattern
        self.g = self.target_far_field()

        # stop any annotation that might be ongoing as we only need to record
        # what's happening in e.solve()
        from pyadjoint.tape import pause_annotation, annotate_tape
        annotate = annotate_tape()
        if annotate:
            pause_annotation()

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
            mesh = fd.Mesh("target.msh")
        else:
            obstacle = {
                "shape": "kite",
                "center": (0.1, 0.),
                "scale": 0.3
            }
            mesh = generate_mesh(obstacle, layer, R0, R1, 3, name="target")

        if hasattr(self.Q, "bbox"):
            plot_mesh(mesh, self.Q.bbox)
        else:
            plot_mesh(mesh)

        solver = PMLSolver(mesh, self.k, self.dirs, a1, b1)
        solver.solve()
        u_target = solver.solutions
        plot_field(u_target[0], layer)
        g = self.far_field(u_target, R0, R1)
        return g

    def far_field(self, u_s, R0, R1):
        """
        Evaluate far field of the scattered wave.
        """
        V = u_s[0].function_space()
        mesh = V.mesh()
        y = fd.SpatialCoordinate(mesh)
        k = self.k
        c = 1 / np.sqrt(8 * np.pi) / fd.sqrt(k)

        # cut-off function
        if mesh != self.grad_psi.function_space().mesh():
            r = fd.sqrt(fd.dot(y, y))
            psi = (1 - fd.cos((r - R0) / (R1 - R0) * fd.pi)) / 2
            grad_psi = fd.interpolate(fd.grad(psi), V)
            laplace_psi = fd.interpolate(fd.div(fd.grad(psi)),
                                         fd.FunctionSpace(mesh, "CG", 1))
        else:
            grad_psi = self.grad_psi
            laplace_psi = self.laplace_psi

        # product of complex numbers
        def dot(x, y):
            x_re, x_im = x
            y_re, y_im = y
            res_re = x_re * y_re - x_im * y_im
            res_im = x_re * y_im + x_im * y_re
            return res_re, res_im

        far_fields = []
        fcp = {"quadrature_degree": 4}
        for u in u_s:
            u_inf = []
            for i in range(self.n):
                x = fd.Constant(self.xs[i, :])
                phi = fd.pi / 4 - k * fd.inner(x, y)
                f = (fd.cos(phi), fd.sin(phi))
                g = dot(u, (laplace_psi, -2 * k * fd.dot(x, grad_psi)))
                h = dot(f, g)
                u_inf_re = fd.assemble(c * h[0] * fd.dx(2),
                                       form_compiler_parameters=fcp)
                u_inf_im = fd.assemble(c * h[1] * fd.dx(2),
                                       form_compiler_parameters=fcp)
                u_inf.append((u_inf_re, u_inf_im))
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
        self.u_inf = self.far_field(self.u_s, self.R0, self.R1)

    def derivative(self, out):
        """
        Get the derivative from pyadjoint.
        """
        deriv = self.Jred.derivative()
        out.from_first_derivative(deriv)

        plot_field(self.u_s[0], self.layer)
        plot_far_field(self.u_inf[0], self.g[0])
        if hasattr(self.Q, "bbox"):
            plot_vector_field(deriv, self.layer, self.Q.bbox)
        else:
            plot_vector_field(deriv, self.layer)

    def gradient(self, g, x, tol):
        super().gradient(g, x, tol)

        T = fd.Function(self.Q.T)
        g.to_coordinatefield(T)
        if hasattr(self.Q, "bbox"):
            plot_vector_field(T, self.layer, self.Q.bbox)
        else:
            plot_vector_field(T, self.layer)
