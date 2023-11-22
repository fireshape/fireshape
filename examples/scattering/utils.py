from math import ceil
import gmsh
import numpy as np
import matplotlib.pyplot as plt
import firedrake as fd


def generate_mesh(obstacle, layer, R0, R1, level, name="mesh"):
    """
    Generate the mesh for the physical domain with an absorbing layer.

    Inputs:
    obstacle: parameters of the obstacle.
    layer: [(a0, a1), (b0, b1)],
           position of the vertical and horizontal layer.
    R0, R1: radius of the annular subdomain for far field evaluation.
    level: level of refinement.
    """
    a0, a1 = layer[0]
    b0, b1 = layer[1]
    gmsh.initialize()

    # define absorbing layer
    gmsh.model.geo.addPoint(a0, -b0, 0, 1, 1)
    gmsh.model.geo.addPoint(a0, b0, 0, 1, 2)
    gmsh.model.geo.addPoint(-a0, b0, 0, 1, 3)
    gmsh.model.geo.addPoint(-a0, -b0, 0, 1, 4)
    gmsh.model.geo.addPoint(a1, -b1, 0, 1, 5)
    gmsh.model.geo.addPoint(a1, -b0, 0, 1, 6)
    gmsh.model.geo.addPoint(a1, b0, 0, 1, 7)
    gmsh.model.geo.addPoint(a1, b1, 0, 1, 8)
    gmsh.model.geo.addPoint(a0, b1, 0, 1, 9)
    gmsh.model.geo.addPoint(-a0, b1, 0, 1, 10)
    gmsh.model.geo.addPoint(-a1, b1, 0, 1, 11)
    gmsh.model.geo.addPoint(-a1, b0, 0, 1, 12)
    gmsh.model.geo.addPoint(-a1, -b0, 0, 1, 13)
    gmsh.model.geo.addPoint(-a1, -b1, 0, 1, 14)
    gmsh.model.geo.addPoint(-a0, -b1, 0, 1, 15)
    gmsh.model.geo.addPoint(a0, -b1, 0, 1, 16)

    gmsh.model.geo.addLine(1, 2, 1)
    gmsh.model.geo.addLine(2, 3, 2)
    gmsh.model.geo.addLine(3, 4, 3)
    gmsh.model.geo.addLine(4, 1, 4)
    gmsh.model.geo.addLine(6, 7, 5)
    gmsh.model.geo.addLine(9, 10, 6)
    gmsh.model.geo.addLine(12, 13, 7)
    gmsh.model.geo.addLine(15, 16, 8)
    gmsh.model.geo.addLine(2, 7, 9)
    gmsh.model.geo.addLine(7, 8, 10)
    gmsh.model.geo.addLine(8, 9, 11)
    gmsh.model.geo.addLine(9, 2, 12)
    gmsh.model.geo.addLine(3, 10, 13)
    gmsh.model.geo.addLine(10, 11, 14)
    gmsh.model.geo.addLine(11, 12, 15)
    gmsh.model.geo.addLine(12, 3, 16)
    gmsh.model.geo.addLine(4, 13, 17)
    gmsh.model.geo.addLine(13, 14, 18)
    gmsh.model.geo.addLine(14, 15, 19)
    gmsh.model.geo.addLine(15, 4, 20)
    gmsh.model.geo.addLine(1, 16, 21)
    gmsh.model.geo.addLine(16, 5, 22)
    gmsh.model.geo.addLine(5, 6, 23)
    gmsh.model.geo.addLine(6, 1, 24)

    gmsh.model.geo.addCurveLoop(range(1, 5), 1)
    gmsh.model.geo.addCurveLoop([5, -9, -1, -24], 2)
    gmsh.model.geo.addCurveLoop([-12, 6, -13, -2], 3)
    gmsh.model.geo.addCurveLoop([-3, -16, 7, -17], 4)
    gmsh.model.geo.addCurveLoop([-21, -4, -20, 8], 5)
    gmsh.model.geo.addCurveLoop(range(9, 13), 6)
    gmsh.model.geo.addCurveLoop(range(13, 17), 7)
    gmsh.model.geo.addCurveLoop(range(17, 21), 8)
    gmsh.model.geo.addCurveLoop(range(21, 25), 9)

    for cl in range(1, 9):
        gmsh.model.geo.addPlaneSurface([cl + 1], cl)

    # define obstacle
    shape = obstacle.get("shape", "circle")
    if shape == "circle":
        x, y = obstacle.get("shift", (0, 0))
        r = obstacle.get("scale", 1)
        p0 = gmsh.model.geo.addPoint(x, y, 0, 1)
        p1 = gmsh.model.geo.addPoint(x + r, y, 0, 1)
        p2 = gmsh.model.geo.addPoint(x, y + r, 0, 1)
        p3 = gmsh.model.geo.addPoint(x - r, y, 0, 1)
        p4 = gmsh.model.geo.addPoint(x, y - r, 0, 1)
        cs = []  # tags of boundary curves
        cs.append(gmsh.model.geo.addCircleArc(p1, p0, p2))
        cs.append(gmsh.model.geo.addCircleArc(p2, p0, p3))
        cs.append(gmsh.model.geo.addCircleArc(p3, p0, p4))
        cs.append(gmsh.model.geo.addCircleArc(p4, p0, p1))

    elif shape == "kite":
        x, y = obstacle.get("shift", (0, 0))
        scale = obstacle.get("scale", 1)
        n = obstacle.get("nodes", 50)  # number of boundary nodes
        ts = np.linspace(0, 2 * np.pi, n, endpoint=False)
        xs = scale * (np.cos(ts) + 0.65 * np.cos(2 * ts) - 0.65) + x
        ys = scale * 1.5 * np.sin(ts) + y

        ps = []  # tags of boundary nodes
        for i in range(ts.size):
            ps.append(gmsh.model.geo.addPoint(xs[i], ys[i], 0, 1))

        cs = []  # tags of boundary curves
        for i in range(n - 1):
            cs.append(gmsh.model.geo.addLine(ps[i], ps[i + 1]))
        cs.append(gmsh.model.geo.addLine(ps[n - 1], ps[0]))

    elif shape == "square":
        x, y = obstacle.get("shift", (0, 0))
        L = obstacle.get("scale", 1)
        p0 = gmsh.model.geo.addPoint(x + L/2, y + L/2, 0, 1)
        p1 = gmsh.model.geo.addPoint(x - L/2, y + L/2, 0, 1)
        p2 = gmsh.model.geo.addPoint(x - L/2, y - L/2, 0, 1)
        p3 = gmsh.model.geo.addPoint(x + L/2, y - L/2, 0, 1)
        cs = []  # tags of boundary curves
        cs.append(gmsh.model.geo.addLine(p0, p1))
        cs.append(gmsh.model.geo.addLine(p1, p2))
        cs.append(gmsh.model.geo.addLine(p2, p3))
        cs.append(gmsh.model.geo.addLine(p3, p0))

    else:
        print("Unsupported shape.")
        raise NotImplementedError

    cl = gmsh.model.geo.addCurveLoop(cs)

    # define subdomain for far field evaluation
    p0 = gmsh.model.geo.addPoint(0, 0, 0, 1)
    p1 = gmsh.model.geo.addPoint(R0, 0, 0, 1)
    p2 = gmsh.model.geo.addPoint(0, R0, 0, 1)
    p3 = gmsh.model.geo.addPoint(-R0, 0, 0, 1)
    p4 = gmsh.model.geo.addPoint(0, -R0, 0, 1)
    cs0 = []
    cs0.append(gmsh.model.geo.addCircleArc(p1, p0, p2))
    cs0.append(gmsh.model.geo.addCircleArc(p2, p0, p3))
    cs0.append(gmsh.model.geo.addCircleArc(p3, p0, p4))
    cs0.append(gmsh.model.geo.addCircleArc(p4, p0, p1))
    cl0 = gmsh.model.geo.addCurveLoop(cs0)

    p0 = gmsh.model.geo.addPoint(0, 0, 0, 1)
    p1 = gmsh.model.geo.addPoint(R1, 0, 0, 1)
    p2 = gmsh.model.geo.addPoint(0, R1, 0, 1)
    p3 = gmsh.model.geo.addPoint(-R1, 0, 0, 1)
    p4 = gmsh.model.geo.addPoint(0, -R1, 0, 1)
    cs1 = []
    cs1.append(gmsh.model.geo.addCircleArc(p1, p0, p2))
    cs1.append(gmsh.model.geo.addCircleArc(p2, p0, p3))
    cs1.append(gmsh.model.geo.addCircleArc(p3, p0, p4))
    cs1.append(gmsh.model.geo.addCircleArc(p4, p0, p1))
    cl1 = gmsh.model.geo.addCurveLoop(cs1)

    gmsh.model.geo.addPlaneSurface([cl, cl0], 9)
    gmsh.model.geo.addPlaneSurface([cl0, cl1], 10)
    gmsh.model.geo.addPlaneSurface([cl1, 1], 11)

    # set cell size
    N = 2  # number of rectangles through thickness of absorbing layer
    NN = ceil(2 * max(a0, b0) / min(a1 - a0, b1 - b0)) * N
    for c in range(1, 9):
        gmsh.model.geo.mesh.setTransfiniteCurve(c, NN + 1)
    for s in range(1, 5):
        gmsh.model.geo.mesh.setTransfiniteSurface(s)

    h0 = min(a1 - a0, b1 - b0) / N  # cell size of physical domain
    gmsh.option.setNumber("Mesh.MeshSizeFactor", h0)

    gmsh.model.geo.synchronize()

    # set subdomain ids
    gmsh.model.addPhysicalGroup(1, cs, 1, name="Gamma")
    gmsh.model.addPhysicalGroup(1, cs0, 2, name="R0")
    gmsh.model.addPhysicalGroup(1, cs1, 3, name="R1")
    gmsh.model.addPhysicalGroup(1, range(1, 5), 4, name="Gamma_I")
    gmsh.model.addPhysicalGroup(
        1, list(range(5, 9)) + [10, 11, 14, 15, 18, 19, 22, 23], 5,
        name="Gamma_D")

    gmsh.model.addPhysicalGroup(2, [9, 11], 1, name="Omega_F1")
    gmsh.model.addPhysicalGroup(2, [10], 2, name="Omega_F2")
    gmsh.model.addPhysicalGroup(2, [1, 3], 3, name="Omega_Ax")
    gmsh.model.addPhysicalGroup(2, [2, 4], 4, name="Omega_Ay")
    gmsh.model.addPhysicalGroup(2, range(5, 9), 5, name="Omega_Axy")

    # generate mesh
    gmsh.model.mesh.generate(2)
    for _ in range(level):
        gmsh.model.mesh.refine()

    msh_file = name + ".msh"
    gmsh.write(msh_file)

    gmsh.finalize()

    return fd.Mesh(msh_file, name=name)


def plot_mesh(mesh, bbox=None, name=None):
    fig, ax = plt.subplots()
    interior_kw = {"linewidths": 0.2}
    fd.triplot(mesh, axes=ax, interior_kw=interior_kw)
    ax.set_aspect("equal")
    ax.legend()

    if bbox:
        ax.add_patch(plt.Rectangle((bbox[0][0], bbox[1][0]),
                                   bbox[0][1] - bbox[0][0],
                                   bbox[1][1] - bbox[1][0],
                                   color='b', fill=False))

    if name:
        plt.savefig(name + ".png", dpi=150, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()


def plot_field(u, layer, name=None):
    a0, a1 = layer[0]
    b0, b1 = layer[1]
    fig, (ax1, ax2) = plt.subplots(1, 2, constrained_layout=True)

    plot1 = fd.tripcolor(u.sub(0), axes=ax1)
    ax1.set_aspect("equal")
    ax1.set_title("Real part")
    ax1.set_xlim(-a1, a1)
    ax1.set_ylim(-b1, b1)
    ax1.add_patch(
        plt.Rectangle((-a0, -b0), 2*a0, 2*b0, color='w', fill=False))
    fig.colorbar(plot1, shrink=0.5, ax=ax1)

    plot2 = fd.tripcolor(u.sub(1), axes=ax2)
    ax2.set_aspect("equal")
    ax2.set_title("Imaginary part")
    ax2.set_xlim(-a1, a1)
    ax2.set_ylim(-b1, b1)
    ax2.add_patch(
        plt.Rectangle((-a0, -b0), 2*a0, 2*b0, color='w', fill=False))
    fig.colorbar(plot2, shrink=0.5, ax=ax2)

    if name:
        plt.savefig(name + ".png", dpi=150, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()


def plot_vector_field(v, layer, bbox=None, name=None):
    a0, a1 = layer[0]
    b0, b1 = layer[1]
    fig, (ax1, ax2) = plt.subplots(1, 2, constrained_layout=True)

    plot1 = fd.tripcolor(v.sub(0), axes=ax1)
    ax1.set_aspect("equal")
    ax1.set_title("x-component")
    ax1.set_xlim(-a1, a1)
    ax1.set_ylim(-b1, b1)
    ax1.add_patch(
        plt.Rectangle((-a0, -b0), 2*a0, 2*b0, color='w', fill=False))
    fig.colorbar(plot1, shrink=0.5, ax=ax1)

    plot2 = fd.tripcolor(v.sub(1), axes=ax2)
    ax2.set_aspect("equal")
    ax2.set_title("y-component")
    ax2.set_xlim(-a1, a1)
    ax2.set_ylim(-b1, b1)
    ax2.add_patch(
        plt.Rectangle((-a0, -b0), 2*a0, 2*b0, color='w', fill=False))
    fig.colorbar(plot2, shrink=0.5, ax=ax2)

    if bbox:
        for ax in ax1, ax2:
            ax.add_patch(plt.Rectangle((bbox[0][0], bbox[1][0]),
                                       bbox[0][1] - bbox[0][0],
                                       bbox[1][1] - bbox[1][0],
                                       color='w', fill=False))

    if name:
        plt.savefig(name + ".png", dpi=150, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()


def plot_far_field(u_inf, g, name=None):
    n = len(u_inf)
    theta = 2 * np.pi / n * np.arange(n + 1)
    u_inf = np.array(u_inf + [u_inf[0]])
    g = np.array(g + [g[0]])
    fig, (ax1, ax2) = plt.subplots(
        1, 2, subplot_kw={'projection': 'polar'}, constrained_layout=True)

    ax1.plot(theta, u_inf[:, 0], label=r"$u_\infty$")
    ax1.plot(theta, g[:, 0], label="target")
    ax1.set_title("Real part")
    ax1.set_rlabel_position(90)
    ax1.grid(True)
    ax1.legend()

    ax2.plot(theta, u_inf[:, 1], label=r"$u_\infty$")
    ax2.plot(theta, g[:, 1], label="target")
    ax2.set_title("Imaginary part")
    ax2.set_rlabel_position(90)
    ax2.grid(True)
    ax2.legend()

    if name:
        plt.savefig(name + ".png", dpi=150, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()
