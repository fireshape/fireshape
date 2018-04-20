from firedrake import Mesh, COMM_WORLD, COMM_SELF
import time
import os
from sys import platform
from subprocess import call


def mesh_from_gmsh_code(geo_code, clscale=1.0, dim=2, comm=COMM_WORLD,
                        name="temp", delete_files=True):
    if comm.rank == 0:
        with open("%s.geo" % name, "w") as text_file:
            text_file.write(geo_code)
    comm.Barrier()
    generateGmsh("%s.geo" % name, "%s.msh" % name, dim, clscale,
                 comm=comm)
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


def generateGmsh(inputFile, outputFile, dimension, scale, comm=COMM_WORLD):
    if platform == "linux" or platform == "linux2":
        if comm.size == 1:
            call(["gmsh", inputFile, "-o", outputFile, "-%i" % dimension,
                  "-clscale", "%f" % scale])
        else:
            if comm.rank == 0:
                """
                Extreme ugly work-around (see https://code.launchpad.net/
                ~fluidity-core/fluidity/firedrake-use-gmshpy/+merge/185785)
                """
                COMM_SELF.Spawn('gmsh', args=[
                    inputFile, "-o", outputFile,
                    "-%i" % dimension, "-clscale", "%f" % scale
                    ])
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
            os.system("gmsh %s -o %s -%i -clscale %f" % (
                inputFile, outputFile, dimension, scale
                ))
        comm.Barrier()
    else:
        raise SystemError("What are you using if not linux or macOS?!")
