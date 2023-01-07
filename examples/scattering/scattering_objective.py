import firedrake as fd
from fireshape import ShapeObjective
import numpy as np
from scattering_PDEconstraint import PMLSolver
from utils import generate_mesh


class FarFieldObjective(ShapeObjective):
    """Misfit of the far field pattern."""

    def __init__(self, pde_solver: PMLSolver, R0, R1, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.R0 = R0
        self.R1 = R1
        self.e = pde_solver
        self.k = pde_solver.k
        self.d = pde_solver.d
        self.u_s = pde_solver.solution
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

        if use_cached_mesh:
            mesh = fd.Mesh("target.msh")
        else:
            obstacle = {
                "shape": "circle",
                "center": (0.1, 0.0),
                "scale": 0.5,
                "nodes": 50
            }
            h0 = 0.1
            mesh = generate_mesh(
                a0, a1, b0, b1, R0, R1, obstacle, h0, name="target")

        solver = PMLSolver(mesh, self.k, self.d, a1, b1)
        solver.solve()
        u_target = solver.solution
        fd.File("u_target.pvd").write(u_target.sub(0))
        return self.far_field(u_target, R0, R1)

    def far_field(self, u_s, R0, R1):
        """
        Evaluate far field of the scattered wave.
        """
        # product of complex numbers
        def dot(x, y):
            x_re, x_im = x
            y_re, y_im = y
            res_re = x_re * y_re - x_im * y_im
            res_im = x_re * y_im + x_im * y_re
            return res_re, res_im

        mesh = u_s.function_space().mesh()
        y = fd.SpatialCoordinate(mesh)
        k = fd.Constant(self.k)
        c = 1 / np.sqrt(8 * np.pi * self.k)

        # cut-off function
        r = fd.sqrt(fd.dot(y, y))
        psi = (1 - fd.cos((r - R0) / (R1 - R0) * fd.pi)) / 2

        theta = np.linspace(0, 2 * np.pi, 100)
        u_inf = []
        fcp = {"quadrature_degree": 4}
        for t in theta:
            x = fd.Constant((np.cos(t), np.sin(t)))
            phi = fd.pi / 4 - k * fd.inner(x, y)
            f = (fd.cos(phi), fd.sin(phi))
            g = dot(u_s, (fd.div(fd.grad(psi)),
                          -2 * k * fd.dot(x, fd.grad(psi))))
            h = dot(f, g)
            u_inf_re = fd.assemble(h[0] * fd.dx(2),
                                   form_compiler_parameters=fcp)
            u_inf_im = fd.assemble(h[1] * fd.dx(2),
                                   form_compiler_parameters=fcp)
            u_inf.append((u_inf_re * c, u_inf_im * c))
        # self.plot_far_field(u_inf)
        return u_inf

    def plot_far_field(self, u_inf):
        import matplotlib.pyplot as plt
        theta = np.linspace(0, 2 * np.pi, len(u_inf))
        u_inf = np.array(u_inf)
        fig, (ax1, ax2) = plt.subplots(
            1, 2, subplot_kw={'projection': 'polar'}, constrained_layout=True)
        ax1.plot(theta, u_inf[:, 0])
        ax1.set_title("Real part")
        ax1.set_rlabel_position(90)
        ax1.grid(True)
        ax2.plot(theta, u_inf[:, 1])
        ax2.set_title("Imaginary part")
        ax2.set_rlabel_position(90)
        ax2.grid(True)
        plt.show()

    def value(self, x, tol):
        """
        Evaluate misfit functional.
        Function signature imposed by ROL.
        """
        u_inf = self.far_field(self.u_s, self.R0, self.R1)
        n = len(u_inf)
        res = 0
        for i in range(n):
            res += (u_inf[i][0] - self.g[i][0])**2 +\
                   (u_inf[i][1] - self.g[i][1])**2
        return res * 2 * np.pi / n

    def derivative(self, out):
        """
        Get the derivative from pyadjoint.
        """
        out.from_first_derivative(self.Jred.derivative())

    def update(self, x, flag, iteration):
        """
        Update domain and solution to state and adjoint equation.
        """
        if self.Q.update_domain(x):
            try:
                # We use pyadjoint to calculate adjoint and shape derivatives,
                # in order to do this we need to "record a tape of the forward
                # solve", pyadjoint will then figure out all necessary
                # adjoints.
                import firedrake_adjoint as fda
                tape = fda.get_working_tape()
                tape.clear_tape()
                # ensure we are annotating
                from pyadjoint.tape import annotate_tape
                safety_counter = 0
                while not annotate_tape():
                    safety_counter += 1
                    fda.continue_annotation()
                    if safety_counter > 1e2:
                        import sys
                        sys.exit('Cannot annotate even after 100 attempts.')
                mesh_m = self.Q.mesh_m
                s = fd.Function(self.V_m)
                mesh_m.coordinates.assign(mesh_m.coordinates + s)
                self.s = s
                self.c = fda.Control(s)
                self.e.solve()
                Jpyadj = self.value(x, None)
                self.Jred = fda.ReducedFunctional(Jpyadj, self.c)
                fda.pause_annotation()
            except fd.ConvergenceError:
                if self.cb is not None:
                    self.cb()
                raise
        if iteration >= 0 and self.cb is not None:
            self.cb()
