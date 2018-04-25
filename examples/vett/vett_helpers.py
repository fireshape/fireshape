import numpy as np
from firedrake import DirichletBC, utils, Constant, Function, \
    SpatialCoordinate, File, prolong, FunctionSpace
import os
import shutil


class NodeDirichletBC(DirichletBC):
    def __init__(self, V, g, nodelist):
        self.nodelist = nodelist
        DirichletBC.__init__(self, V, g, None)

    @utils.cached_property
    def nodes(self):
        return self.nodelist


def get_extra_bc(V_control, lower, upper):
    X = SpatialCoordinate(V_control.mesh())
    coords = Function(V_control).interpolate(X)
    coords = coords.vector()[:, 0]
    nodes = np.where((coords >= upper-1e-6) | (coords <= lower+1e-6))[0]
    return NodeDirichletBC(V_control, Constant((0., 0.)), nodes)


class NavierStokesWriter(object):

    def __init__(self, outdir, s, comm, f_mesh=None):
        if comm.rank == 0:
            if os.path.exists(outdir):
                shutil.rmtree(outdir)
            os.makedirs(outdir)
        self.outdir = outdir
        self.s = s
        self.solution = s.solution
        self.solution_adj = s.solution_adj
        self.pvdu = File(f"{outdir}u.pvd")
        self.pvdp = File(f"{outdir}p.pvd")
        self.pvdv = File(f"{outdir}v.pvd")
        self.pvdq = File(f"{outdir}q.pvd")
        self.f_mesh = f_mesh
        if f_mesh is not None:
            element = self.solution.function_space().ufl_element()
            V = FunctionSpace(f_mesh, element)
            self.f_solution = Function(V)
            self.f_solution_adj = Function(V)
            self.pvdfu = File(f"{outdir}fu.pvd")
            self.pvdfp = File(f"{outdir}fp.pvd")
            self.pvdfv = File(f"{outdir}fv.pvd")
            self.pvdfq = File(f"{outdir}fq.pvd")

    def write(self):
        self.coarse_write()
        self.fine_write()

    def coarse_write(self):
        u, p = self.solution.split()
        self.pvdu.write(u)
        self.pvdp.write(p)
        v, q = self.solution_adj.split()
        self.pvdv.write(v)
        self.pvdq.write(q)

    def fine_write(self):
        if self.f_mesh is not None:
            u, p = self.solution.split()
            v, q = self.solution_adj.split()
            fu, fp = self.f_solution.split()
            fv, fq = self.f_solution_adj.split()
            prolong(u.ufl_domain().coordinates, self.f_mesh.coordinates)
            prolong(u, fu)
            prolong(p, fp)
            prolong(v, fv)
            prolong(q, fq)
            self.pvdfu.write(fu)
            self.pvdfp.write(fp)
            self.pvdfv.write(fv)
            self.pvdfq.write(fq)
