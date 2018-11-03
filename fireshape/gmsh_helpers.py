from firedrake import Mesh, COMM_WORLD, COMM_SELF
import time
import os
from sys import platform
from subprocess import call


def mesh_from_gmsh_code(geo_code, clscale=1.0, dim=2, comm=COMM_WORLD,
                        name="/tmp/tmp", smooth=1, delete_files=True):
    if comm.rank == 0:
        with open("%s.geo" % name, "w") as text_file:
            text_file.write(geo_code)
    comm.Barrier()
    generateGmsh("%s.geo" % name, "%s.msh" % name, dim, clscale,
                 comm=comm, smooth=smooth)
    mesh = Mesh("%s.msh" % name)
    if delete_files:
        try:
            os.remove("%s.geo" % name)
        except OSError:
            pass
        try:
            os.remove("%s.msh" % name)
        except OSError:
            pass
    return mesh


def generateGmsh(inputFile, outputFile, dimension, scale, comm=COMM_WORLD,
                 smooth=1):
    if platform == "linux" or platform == "linux2":
        if comm.size == 1:
            call(["gmsh", inputFile, "-o", outputFile, "-%i" % dimension,
                  "-clscale", "%f" % scale, "-smooth", "%i" % smooth])
        else:
            if comm.rank == 0:
                """
                Extreme ugly work-around (see https://code.launchpad.net/
                ~fluidity-core/fluidity/firedrake-use-gmshpy/+merge/185785)
                """
                call(["gmsh", inputFile, "-o", outputFile, "-%i" % dimension,
                      "-clscale", "%f" % scale, "-smooth", "%i" % smooth])
                # comm.Spawn('gmsh', args=[
                #     inputFile, "-o", outputFile,
                #     "-%i" % dimension, "-clscale", "%f" % scale#, "-smooth",
                #     # "%i" % smooth
                #     ])
                oldsize = 0
                time.sleep(2)
                while True:
                    try:
                        statinfo = os.stat(outputFile)
                        newsize = statinfo.st_size
                        if newsize == 0 or newsize != oldsize:
                            oldsize = newsize
                            time.sleep(2)
                        else:
                            break
                    except OSError as e:
                        if e.errno == 2:
                            pass
                        else:
                            raise e
            comm.Barrier()
    elif platform == "darwin":
        if comm.rank == 0:
            os.system("gmsh %s -o %s -%i -clscale %f -smooth %i" % (
                inputFile, outputFile, dimension, scale, smooth
                ))
        comm.Barrier()
    else:
        raise SystemError("What are you using if not linux or macOS?!")

def DiskMesh(clscale, radius=1.):
    geo_code = """
Point(1) = {-0, 0, 0, 1.0};
Point(2) = {%f, 0, 0, 1.0};
Point(3) = {-%f, 0, 0, 1.0};
Circle(4) = {2, 1, 3};
Circle(5) = {3, 1, 2};
Line Loop(6) = {4, 5};
Plane Surface(7) = {6};
Physical Line("Boundary") = {6};
Physical Surface("Disk") = {7};
    """ % ((radius,)*2)
    return mesh_from_gmsh_code(geo_code, clscale=clscale, dim=2)


def SphereMesh(clscale, radius=1.):
    geo_code="""
Point(11) = {0, 0, 0, 1.0};
Point(12) = {%f, 0, 0, 1.0};
Point(13) = {-%f, 0, 0, 1.0};
Point(14) = {0, %f, 0, 1.0};
Point(15) = {0, -%f, 0, 1.0};
Point(16) = {0, 0, %f, 1.0};
Point(17) = {0, 0, -%f, 1.0};

Circle(9) = {13, 11, 15};
Circle(10) = {15, 11, 12};
Circle(11) = {12, 11, 14};
Circle(12) = {14, 11, 13};
Circle(13) = {13, 11, 16};
Circle(14) = {16, 11, 12};
Circle(15) = {12, 11, 17};
Circle(16) = {17, 11, 13};
Circle(17) = {17, 11, 14};
Circle(18) = {14, 11, 16};
Circle(19) = {16, 11, 15};
Circle(20) = {15, 11, 17};

Line Loop(37) = {19, 10, -14};
Ruled Surface(38) = {37};
Line Loop(39) = {15, -20, 10};
Ruled Surface(40) = {39};
Line Loop(41) = {16, 9, 20};
Ruled Surface(42) = {41};
Line Loop(43) = {13, 19, -9};
Ruled Surface(44) = {43};
Line Loop(45) = {15, 17, -11};
Ruled Surface(46) = {45};
Line Loop(47) = {11, 18, 14};
Ruled Surface(48) = {47};
Line Loop(49) = {18, -13, -12};
Ruled Surface(50) = {49};
Line Loop(51) = {17, 12, -16};
Ruled Surface(52) = {51};

Surface Loop(54) = {50, 48, 46, 40, 42, 52, 44, 38};
Volume(55) = {54};
Physical Surface("Surface") = {54};
Physical Volume("Sphere") = {55};
""" % ((radius, ) * 6)
    return mesh_from_gmsh_code(geo_code, clscale=clscale, dim=3)
