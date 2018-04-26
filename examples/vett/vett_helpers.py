import numpy as np
from firedrake import DirichletBC, utils, Constant, Function, \
    SpatialCoordinate, File, prolong, FunctionSpace
import os
import shutil
import csv


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
        comm.Barrier()
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


def collect_1d_arrays(local_array, comm):
    rank = comm.rank
    root = 0
    sendbuf = np.array(local_array)
    sendcounts = np.array(comm.gather(len(sendbuf), root))
    if rank == root:
        recvbuf = np.empty((sum(sendcounts), 1), dtype=local_array.dtype)
    else:
        recvbuf = None
    comm.Gatherv(sendbuf=sendbuf, recvbuf=(recvbuf, sendcounts), root=root)
    return recvbuf


def collect_2d_arrays(local_array, comm):
    return np.hstack([collect_1d_arrays(local_array[:, i], comm) for i in range(local_array.shape[1])])


def get_boundary_coords(mesh):
    coords = []
    vec = mesh.coordinates.vector()
    markers = mesh.topological.exterior_facets.unique_markers
    for marker in markers:
        bc = DirichletBC(mesh.coordinates.function_space(), 0, int(marker))
        nodes = bc.nodes
        nodes = nodes[nodes < vec.local_size()]
        coords.append(collect_2d_arrays(vec[nodes, :], mesh.comm))
    return coords


def export_boundary(mesh, outdir):
    coords = get_boundary_coords(mesh)
    if mesh.comm.rank == 0:
        for i in range(len(coords)):
            with open(f"{outdir}coords_boundary_{i}.csv", "w") as csvfile:
                writer = csv.writer(csvfile, delimiter=",")
                for x in coords[i]:
                    writer.writerow(x)
    mesh.comm.Barrier()
