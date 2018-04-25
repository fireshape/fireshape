import pygmsh
import numpy as np


def create_rounded_diffusor(xvals, hvals):
    geom = pygmsh.built_in.Geometry()
    fak = 0.7

    l = len(xvals)  # noqa
    top_points = []
    for i in range(l):
        lcar = 0.06 * fak
        if i == 0:
            lcar *= 0.3
        x = np.asarray([xvals[i], hvals[i], 0.])
        top_points.append(geom.add_point(x=x, lcar=lcar))
    bottom_points = []
    for i in range(l):
        bottom_points.append(geom.add_point(x=np.asarray([xvals[i], 0., 0.]),
                                            lcar=1.0 * fak))

    top_lines = [geom.add_line(top_points[0], top_points[1])]
    top_lines.append(geom.add_bspline(top_points[1:l-1]))
    top_lines.append(geom.add_line(top_points[l-2], top_points[l-1]))

    bottom_lines = [geom.add_line(bottom_points[0], bottom_points[1])]
    bottom_lines.append(geom.add_line(bottom_points[1], bottom_points[l-2]))
    bottom_lines.append(geom.add_line(bottom_points[l-2], bottom_points[l-1]))

    vert_lines = []
    for i in [0, 1, l-2, l-1]:
        vert_lines.append(geom.add_line(bottom_points[i], top_points[i]))

    loops = []
    for i in range(3):
        loops.append(geom.add_line_loop([bottom_lines[i], vert_lines[i+1],
                                         -top_lines[i], -vert_lines[i]]))

    plane_surfaces = []
    for i in range(3):
        plane_surfaces.append(geom.add_plane_surface(loops[i]))

    geom.add_physical_line(vert_lines[0], label="Inflow")
    geom.add_physical_line([top_lines[0], top_lines[-1]], label="NoSlipFixed")
    geom.add_physical_line(top_lines[1:-1], label="NoSlipFree")
    geom.add_physical_line(vert_lines[-1], label="Outflow")
    geom.add_physical_line(bottom_lines, label="Symmetry")
    geom.add_physical_surface(plane_surfaces, label="Channel")

    return geom.get_code()


if __name__ == "__main__":
    xvals = [0.0, 0.9, 1.0, 3.9, 4.0, 5.0]
    hvals = [1.0, 1.0, 1.0, 0.7, 0.7, 0.7]
    create_rounded_diffusor(xvals, hvals)
