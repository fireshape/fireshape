ifeq ($(GMSH),)
GMSH := gmsh
endif

3d:
	echo Tested with gmsh v4.1.2. To specify a gmsh binary to use, simple \'export GMSH=/location/of/gmsh\'
	${GMSH} -3 -clscale 0.2 -format msh2 -o pipe.msh pipe3d.geo

2d:
	echo Tested with gmsh v4.1.2. To specify a gmsh binary to use, simple \'export GMSH=/location/of/gmsh\'
	${GMSH} -2 -clscale 0.2 -format msh2 -o pipe.msh pipe2d.geo
