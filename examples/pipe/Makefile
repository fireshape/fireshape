ifeq ($(GMSH),)
GMSH := gmsh
endif

all:
	echo "Please run either make 2d or make 3d to obtain the respective mesh."

3d:
	echo Tested with gmsh v4.1.2. To specify a gmsh binary to use, simple \'export GMSH=/location/of/gmsh\'
	${GMSH} -3 -clscale 0.15 -format msh2 -o pipe.msh pipe3d.geo

2d:
	echo Tested with gmsh v4.1.2. To specify a gmsh binary to use, simple \'export GMSH=/location/of/gmsh\'
	${GMSH} -2 -clscale 0.1 -format msh2 -o pipe.msh pipe2d.geo
