import gmsh
import numpy as np
import firedrake as fd


def generate_mesh(a0, a1, b0, b1, R0, R1, params, h0, name="mesh"):
    """
    Generate the mesh for the physical domain with an abosorbing layer.

    Inputs:
    a0, a1: position of the vertical layer.
    b0, b1: position of the horizontal layer.
    R0, R1: radius of the annular subdomain for far field evaluation.
    params: parameters of the obstacle.
    h0: cell size.
    """
    gmsh.initialize()

    # define absorbing layer
    gmsh.model.occ.addPoint(a0, -b0, 0, 1, 1)
    gmsh.model.occ.addPoint(a0, b0, 0, 1, 2)
    gmsh.model.occ.addPoint(-a0, b0, 0, 1, 3)
    gmsh.model.occ.addPoint(-a0, -b0, 0, 1, 4)
    gmsh.model.occ.addPoint(a1, -b1, 0, 1, 5)
    gmsh.model.occ.addPoint(a1, -b0, 0, 1, 6)
    gmsh.model.occ.addPoint(a1, b0, 0, 1, 7)
    gmsh.model.occ.addPoint(a1, b1, 0, 1, 8)
    gmsh.model.occ.addPoint(a0, b1, 0, 1, 9)
    gmsh.model.occ.addPoint(-a0, b1, 0, 1, 10)
    gmsh.model.occ.addPoint(-a1, b1, 0, 1, 11)
    gmsh.model.occ.addPoint(-a1, b0, 0, 1, 12)
    gmsh.model.occ.addPoint(-a1, -b0, 0, 1, 13)
    gmsh.model.occ.addPoint(-a1, -b1, 0, 1, 14)
    gmsh.model.occ.addPoint(-a0, -b1, 0, 1, 15)
    gmsh.model.occ.addPoint(a0, -b1, 0, 1, 16)

    gmsh.model.occ.addLine(1, 2, 1)
    gmsh.model.occ.addLine(2, 3, 2)
    gmsh.model.occ.addLine(3, 4, 3)
    gmsh.model.occ.addLine(4, 1, 4)
    gmsh.model.occ.addLine(6, 7, 5)
    gmsh.model.occ.addLine(9, 10, 6)
    gmsh.model.occ.addLine(12, 13, 7)
    gmsh.model.occ.addLine(15, 16, 8)
    gmsh.model.occ.addLine(2, 7, 9)
    gmsh.model.occ.addLine(7, 8, 10)
    gmsh.model.occ.addLine(8, 9, 11)
    gmsh.model.occ.addLine(9, 2, 12)
    gmsh.model.occ.addLine(3, 10, 13)
    gmsh.model.occ.addLine(10, 11, 14)
    gmsh.model.occ.addLine(11, 12, 15)
    gmsh.model.occ.addLine(12, 3, 16)
    gmsh.model.occ.addLine(4, 13, 17)
    gmsh.model.occ.addLine(13, 14, 18)
    gmsh.model.occ.addLine(14, 15, 19)
    gmsh.model.occ.addLine(15, 4, 20)
    gmsh.model.occ.addLine(1, 16, 21)
    gmsh.model.occ.addLine(16, 5, 22)
    gmsh.model.occ.addLine(5, 6, 23)
    gmsh.model.occ.addLine(6, 1, 24)

    gmsh.model.occ.addCurveLoop(range(1, 5), 1)
    gmsh.model.occ.addCurveLoop([5, -9, -1, -24], 2)
    gmsh.model.occ.addCurveLoop([-12, 6, -13, -2], 3)
    gmsh.model.occ.addCurveLoop([-3, -16, 7, -17], 4)
    gmsh.model.occ.addCurveLoop([-21, -4, -20, 8], 5)
    gmsh.model.occ.addCurveLoop(range(9, 13), 6)
    gmsh.model.occ.addCurveLoop(range(13, 17), 7)
    gmsh.model.occ.addCurveLoop(range(17, 21), 8)
    gmsh.model.occ.addCurveLoop(range(21, 25), 9)

    for cl in range(1, 9):
        gmsh.model.occ.addPlaneSurface([cl + 1], cl)

    # define obstacle
    shape = params.get("shape", "circle")
    if shape == "circle":
        x, y = params.get("center", (0, 0))
        r = params.get("scale", 1)
        kc = [gmsh.model.occ.addCircle(x, y, 0, r)]  # tag of circle

    elif shape == "kite":
        x, y = params.get("center", (0, 0))
        scale = params.get("scale", 1)
        n = params.get("nodes", 100)  # number of boundary nodes
        ts = np.linspace(0, 2 * np.pi, n, endpoint=False)
        xs = scale * (np.cos(ts) + 0.65 * np.cos(2 * ts) - 0.65) + x
        ys = scale * 1.5 * np.sin(ts) + y

        kp = []  # tags of boundary nodes
        for i in range(ts.size):
            kp.append(gmsh.model.occ.addPoint(xs[i], ys[i], 0, 1))

        kc = []  # tags of boundary segments
        for i in range(n - 1):
            kc.append(gmsh.model.occ.addLine(kp[i], kp[i + 1]))
        kc.append(gmsh.model.occ.addLine(kp[n - 1], kp[0]))

    else:
        print("Unsupported shape.")
        raise NotImplementedError

    cl = gmsh.model.occ.addCurveLoop(kc)

    # define subdomain for far field evaluation
    circ0 = gmsh.model.occ.addCircle(0, 0, 0, R0)
    circ1 = gmsh.model.occ.addCircle(0, 0, 0, R1)

    cl0 = gmsh.model.occ.addCurveLoop([circ0])
    cl1 = gmsh.model.occ.addCurveLoop([circ1])

    gmsh.model.occ.addPlaneSurface([cl, cl0], 9)
    gmsh.model.occ.addPlaneSurface([cl0, cl1], 10)
    gmsh.model.occ.addPlaneSurface([cl1, 1], 11)

    gmsh.model.occ.synchronize()

    # set subdomain ids
    gmsh.model.addPhysicalGroup(1, kc, 1, name="Gamma")
    gmsh.model.addPhysicalGroup(1, [circ0], 2, name="R0")
    gmsh.model.addPhysicalGroup(1, [circ1], 3, name="R1")
    gmsh.model.addPhysicalGroup(1, range(1, 5), 4, name="Gamma_I")
    gmsh.model.addPhysicalGroup(
        1, list(range(5, 9)) + [10, 11, 14, 15, 18, 19, 22, 23], 5,
        name="Gamma_D")

    gmsh.model.addPhysicalGroup(2, [9, 11], 1, name="Omega_F1")
    gmsh.model.addPhysicalGroup(2, [10], 2, name="Omega_F2")
    gmsh.model.addPhysicalGroup(2, [1, 3], 3, name="Omega_Ax")
    gmsh.model.addPhysicalGroup(2, [2, 4], 4, name="Omega_Ay")
    gmsh.model.addPhysicalGroup(2, range(5, 9), 5, name="Omega_Axy")

    # set cell size
    gmsh.option.setNumber("Mesh.MeshSizeFactor", h0)

    # generate mesh
    gmsh.model.mesh.generate(2)
    msh_file = name + ".msh"
    gmsh.write(msh_file)

    gmsh.finalize()

    mesh = fd.Mesh(msh_file)

    import matplotlib.pyplot as plt
    fig, axes = plt.subplots()
    interior_kw = {"linewidths": 0.2}
    fd.triplot(mesh, axes=axes, interior_kw=interior_kw)
    axes.set_aspect("equal")
    axes.legend()
    plt.show()

    return mesh
