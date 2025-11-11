import gmsh
import sys


gmsh.initialize(sys.argv)

gmsh.option.setNumber("General.Terminal", 1)

gmsh.model.add("mesh_cantilever")

# Define Parameters
h = 0.2  # mesh size
disk_radius_left = 0.2
disk_radius_right = 0.2
n_ext = 4  # number of extruded layers

# Set the same size h to every point
gmsh.option.setNumber("Mesh.Algorithm", 6)  # 6: Triangle
gmsh.option.setNumber("Mesh.CharacteristicLengthMin", h)
gmsh.option.setNumber("Mesh.CharacteristicLengthMax", h)

# Define Point entities (at 4 corners)
gmsh.model.occ.addPoint(-1, 0.5, 0, h, 1)
gmsh.model.occ.addPoint(-1, -0.5, 0, h, 2)
gmsh.model.occ.addPoint(1, -0.2, 0, h, 3)
gmsh.model.occ.addPoint(1, 0.2, 0, h, 4)

# Define Curve entities
gmsh.model.occ.addLine(1, 2, 1)
gmsh.model.occ.addLine(2, 3, 2)
gmsh.model.occ.addLine(3, 4, 3)
gmsh.model.occ.addLine(4, 1, 4)

# Define CurveLoop entities
gmsh.model.occ.addCurveLoop([1, 2, 3, 4], 1)

# Define Surface entities
gmsh.model.occ.addPlaneSurface([1], 1)

# Define a Disk in the middle
gmsh.model.occ.addDisk(0, 0, 0, disk_radius_right, disk_radius_right, 2)

# And a Disk on the left edge
gmsh.model.occ.addDisk(-1, 0, 0, disk_radius_left, disk_radius_left, 3)

# Cut holes
gmsh.model.occ.cut([(2, 1)], [(2, 2), (2, 3)], 4)

# Extrude to 3D, can set mesh size separately here
gmsh.model.occ.extrude([(2, 4)], 0, 0, 1, [n_ext])

gmsh.model.occ.synchronize()

# Define PhysicalGroup for boundary domains and volume
gmsh.model.addPhysicalGroup(2, [8, 11], 1)
gmsh.model.addPhysicalGroup(2, [6], 2)
gmsh.model.addPhysicalGroup(3, [1], 100)

gmsh.model.mesh.generate(3)

gmsh.write("mesh_cantilever.msh")

# Open GMSH GUI from Python
gmsh.fltk.run()

gmsh.finalize()
