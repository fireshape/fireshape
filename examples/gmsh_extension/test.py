import pygmsh
import numpy as np
from icecream import ic
import meshio

mesh = meshio.read("stoke_hole.msh")

ic(mesh.points)
ic(mesh.cells)
ic(mesh.cells_dict)