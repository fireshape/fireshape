import pygmsh
import numpy as np


def create_diffusor(xvals, hvals, top_scale=0.06, rounded=True):
    geom = pygmsh.built_in.Geometry()
    fak = 0.7

    l = len(xvals)  # noqa
    top_points = []
    for i in range(l):
        lcar = top_scale * fak
        if i == 0:
            lcar *= 0.3
        x = np.asarray([xvals[i], hvals[i], 0.])
        top_points.append(geom.add_point(x=x, lcar=lcar))
    bottom_points = []
    for i in range(l):
        lcar = fak
        if i == 0:
            lcar *= 0.2
        bottom_points.append(geom.add_point(x=np.asarray([xvals[i], 0., 0.]),
                                            lcar=lcar))

    top_lines = [geom.add_line(top_points[0], top_points[1])]
    if rounded:
        top_lines.append(geom.add_bspline(top_points[1:l-1]))
    else:
        for i in range(1, l-2):
            top_lines.append(geom.add_line(top_points[i], top_points[i+1]))
    top_lines.append(geom.add_line(top_points[l-2], top_points[l-1]))

    bottom_lines = [geom.add_line(bottom_points[0], bottom_points[1])]
    bottom_lines.append(geom.add_line(bottom_points[1], bottom_points[l-2]))
    bottom_lines.append(geom.add_line(bottom_points[l-2], bottom_points[l-1]))

    vert_lines = []
    for i in [0, 1, l-2, l-1]:
        vert_lines.append(geom.add_line(bottom_points[i], top_points[i]))

    loops = []
    loops.append(geom.add_line_loop([bottom_lines[0], vert_lines[1],
                                     -top_lines[0], -vert_lines[0]]))
    loops.append(geom.add_line_loop([bottom_lines[1]] + [vert_lines[-2]] +
                                     [-k for k in reversed(top_lines[1:-1])] + [-vert_lines[1]]))
    loops.append(geom.add_line_loop([bottom_lines[-1], vert_lines[-1],
                                     -top_lines[-1], -vert_lines[-2]]))

    plane_surfaces = []
    for i in range(3):
        plane_surfaces.append(geom.add_plane_surface(loops[i]))

    geom.add_physical_line(vert_lines[0], label="Inflow")
    geom.add_physical_line([top_lines[0], top_lines[-1]], label="NoSlipFixed")
    geom.add_physical_line(top_lines[1:-1], label="NoSlipFree")
    geom.add_physical_line(vert_lines[-1], label="Outflow")
    geom.add_physical_line(bottom_lines, label="Symmetry")
    geom.add_physical_surface(plane_surfaces)

    return geom.get_code()


if __name__ == "__main__":
    xvals = [0.0, 0.9, 1.0, 3.9, 4.0, 5.0]
    hvals = [1.0, 1.0, 1.0, 0.7, 0.7, 0.7]
    code = create_diffusor(xvals, hvals, rounded=True)
    with open("temp.geo", "w") as f:
        f.write(code)
