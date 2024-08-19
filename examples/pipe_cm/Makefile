ifeq ($(GMSH),)
GMSH := gmsh
endif

SCALE := 0.1

2d:
	${GMSH} -2 -clscale ${SCALE} -format msh2 -o pipe.msh pipe2d.geo
	${GMSH} -2 -clscale ${SCALE} -format msh2 -o pipe_control.msh pipe2d_control.geo
