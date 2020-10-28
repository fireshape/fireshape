from firedrake import *
from firedrake.cython import dmcommon
import numpy as np



def mark_mesh(mesh):
    import ipdb; ipdb.set_trace()
    dm = mesh._topology_dm #mesh._plex
    sec = dm.getCoordinateSection()
    coords = dm.getCoordinatesLocal()
    dm.createLabel(dmcommon.FACE_SETS_LABEL)
    nface = dm.getStratumSize("exterior_facets", 1)
    if nface == 0:
        return mesh
    faces = dm.getStratumIS("exterior_facets", 1).indices
    for face in faces:
        #vertices = dm.vecGetClosure(sec, coords, face).reshape(3, 3)
        #in 2D do
        vertices = dm.vecGetClosure(sec, coords, face).reshape(2, 2)
        if np.allclose(vertices[:, 0], 0):
            # left
            dm.setLabelValue(dmcommon.FACE_SETS_LABEL, face, 11)
        elif np.allclose(vertices[:, 0], 1.0):
            # right
            dm.setLabelValue(dmcommon.FACE_SETS_LABEL, face, 12)
        elif np.allclose(vertices[:, 1], 0.0):
            # bottom
            dm.setLabelValue(dmcommon.FACE_SETS_LABEL, face, 13)
        elif np.allclose(vertices[:, 1], 1.0):
            # top
            dm.setLabelValue(dmcommon.FACE_SETS_LABEL, face, 14)
    return mesh

mesh = mark_mesh(PeriodicSquareMesh(10, 10, 1))
#mesh = mark_mesh(UnitSquareMesh(10, 10))
#mesh = UnitSquareMesh(10, 10)
V = FunctionSpace(mesh, "CG", 1)
v = Function(V)
DirichletBC(V, 11, 1).apply(v)
DirichletBC(V, 12, 2).apply(v)
DirichletBC(V, 13, 3).apply(v)
DirichletBC(V, 14, 4).apply(v)
File("test.pvd").write(v)
