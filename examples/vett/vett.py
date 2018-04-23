from diffusor_mesh_rounded import create_rounded_diffusor
from gmsh_helpers import mesh_from_gmsh_code
from firedrake import File, SpatialCoordinate, conditional, ge, \
        sinh, as_vector, DumbCheckpoint, FILE_READ, FILE_CREATE, \
        Function, FunctionSpace, prolong, Constant
from rans_mixing_length import RANSMixingLengthSolver
from vett_objectives import EnergyRecovery
import fireshape as fs
import fireshape.zoo as fsz
import ROL
import numpy as np
import os
import shutil
from mpi4py import MPI
comm = MPI.COMM_WORLD

wokeness = 1.0
inflow_type = 1

xvals = [0.0, 3.0, 9.9, 10.0, 19.9, 20.0, 29.0, 30.0]
hvals = [1.0, 1.0, 1.0, 1.0, 1.5, 1.5, 1.5, 1.5]
mesh_code = create_rounded_diffusor(xvals, hvals)
mesh = mesh_from_gmsh_code(mesh_code, clscale=4.0, smooth=100)
inflow_bids = [1]
noslip_fixed_bids = [2]
noslip_free_bids = [3]
outflow_bids = [4]
symmetry_bids = [5]


def extra_bc_cb(coords):
    return np.where((coords[:, 0] >= 29.-1e-6) | (coords[:, 0] <= 3.+1e-6))[0]


inner = fs.ElasticityInnerProduct(fixed_bids=inflow_bids + noslip_fixed_bids
                                  + outflow_bids + symmetry_bids,
                                  extra_bc_cb=extra_bc_cb)


Q = fs.FeMultiGridControlSpace(mesh, inner, refinements=2)
mesh_m = Q.mesh_m


x, y = SpatialCoordinate(mesh_m)
eps = 0.002
smoother = conditional(ge(y, 1-eps), (1-((1/eps)*(y-(1-eps)))**4)**4, 1.0)


def linear_wake_function(y, wokeness):
    return wokeness * (0.25 + abs(y)*0.75) + (1-wokeness) * (0.5 * 1.25)


def sinh_wake_function(y, R, N):
    def U1(y, N):
        return 1./(1+sinh(y/sinh(1))**(2*N))
    return 1 - R + 2*R*U1(y, N)


if inflow_type == 1:
    xvel = linear_wake_function(y, wokeness)
else:
    xvel = sinh_wake_function(2.0 * y, -wokeness, 2)

inflow_expr = as_vector([smoother * xvel, 0])
solver_t = RANSMixingLengthSolver
Re = 1e6
pvel = 4
s = solver_t(mesh_m, inflow_bids=inflow_bids,
             inflow_expr=inflow_expr,
             noslip_bids=noslip_free_bids + noslip_fixed_bids,
             symmetry_bids=symmetry_bids,
             nu=1.0/Re, velocity_degree=pvel)

(u, p) = s.solution.split()
(v, q) = s.solution_adj.split()
outdir = f"output/"
if comm.rank == 0:
    if not os.path.exists(outdir):
        os.makedirs(outdir)


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


try:
    chk = DumbCheckpoint(f"{outdir}continued_solution", mode=FILE_READ)
    chk.load(s.solution, name="State")
    print("Found continued solution.")
except: # noqa
    writer = NavierStokesWriter(outdir + "continuation/", s, comm)
    s.solve_by_continuation(post_solve_cb=lambda nu: writer.write(), steps=16)
    chk = DumbCheckpoint(f"{outdir}continued_solution", mode=FILE_CREATE)
    chk.store(s.solution, name="State")


outu = File("u.pvd")


def cb():
    outu.write(s.solution.split()[0])


Je = EnergyRecovery(s, Q, scale=1e-1, cb=cb)
Jr = fs.ReducedObjective(Je, s)
Js = fsz.MoYoSpectralConstraint(10., Constant(0.5), Q)
J = Jr + Js
q = fs.ControlVector(Q)
J.update(q, None, 1)
g = fs.ControlVector(Q)
J.gradient(g, q, None)
g.scale(1e2)
J.checkGradient(q, g, 4, 1)

params_dict = {
    'General': {
        'Secant': {'Type': 'Limited-Memory BFGS',
                   'Maximum Storage': 5}},
    'Step': {
        'Type': 'Line Search',
        'Line Search': {'Descent Method': {
            'Type': 'Quasi-Newton Step'}}},
    'Status Test': {
        'Gradient Tolerance': 1e-4,
        'Relative Gradient Tolerance': 1e-3,
        'Step Tolerance': 1e-10, 'Relative Step Tolerance': 1e-10,
        'Iteration Limit': 10}}

params = ROL.ParameterList(params_dict, "Parameters")
problem = ROL.OptimizationProblem(J, q)
solver = ROL.OptimizationSolver(problem, params)
solver.solve()
