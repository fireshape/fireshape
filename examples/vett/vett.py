from diffusor_mesh_rounded import create_rounded_diffusor
from gmsh_helpers import mesh_from_gmsh_code
from firedrake import File


xvals = [0.0, 0.9, 1.0, 3.9, 4.0, 5.0]
hvals = [1.0, 1.0, 1.0, 0.7, 0.7,  0.7, 0.7]
mesh_code = create_rounded_diffusor(xvals, hvals)
mesh = mesh_from_gmsh_code(mesh_code)

File("coords.pvd").write(mesh.coordinates)
