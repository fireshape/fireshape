from diffusor_mesh_rounded import create_rounded_diffusor
from gmsh_helpers import mesh_from_gmsh_code
from firedrake import File, SpatialCoordinate, conditional, ge, \
    sinh, as_vector, DumbCheckpoint, FILE_READ, FILE_CREATE, \
    Constant, MeshHierarchy
from rans_mixing_length import RANSMixingLengthSolver
from vett_objectives import EnergyRecovery
from vett_helpers import get_extra_bc, NavierStokesWriter
import fireshape as fs
import fireshape.zoo as fsz
import ROL
import os
from mpi4py import MPI
comm = MPI.COMM_WORLD

wokeness = 1.0
inflow_type = 1

if True:
    xvals = [0.0, 1.0, 6.9, 7.0, 18.9, 19.0, 29.0, 30.0]
    h2 = 1.5
    hvals = [1.0, 1.0, 1.0, 1.0, h2, h2, h2, h2]
else:
    xvals = [0.0, 1.0, 4.9, 5.0, 7.9, 8.0, 9.0, 10.0]
    h2 = 1./1.5
    hvals = [1.0, 1.0, 1.0, 1.0, h2, h2, h2, h2]
mesh_code = create_rounded_diffusor(xvals, hvals)
mesh = mesh_from_gmsh_code(mesh_code, clscale=1.0, smooth=100)
inflow_bids = [1]
noslip_fixed_bids = [2]
noslip_free_bids = [3]
outflow_bids = [4]
symmetry_bids = [5]


Q = fs.FeMultiGridControlSpace(mesh, refinements=1, order=1)
# Q = fs.FeControlSpace(mesh)
mesh_m = Q.mesh_m
# mesh_f = MeshHierarchy(mesh_m, 2)[-1]

(V_control, _) = Q.get_space_for_inner()
extra_bc = get_extra_bc(V_control, 1., max(xvals)-1.0)
inner = fs.ElasticityInnerProduct(Q, fixed_bids=inflow_bids + noslip_fixed_bids
                                  + outflow_bids + symmetry_bids,
                                  extra_bcs=extra_bc)

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
pvel = 3
s = solver_t(mesh_m, inflow_bids=inflow_bids,
             inflow_expr=inflow_expr,
             noslip_bids=noslip_free_bids + noslip_fixed_bids,
             symmetry_bids=symmetry_bids,
             nu=1.0/Re, velocity_degree=pvel)

(u, p) = s.solution.split()
(v, q) = s.solution_adj.split()
outdir = f"output_5/"
if comm.rank == 0:
    if not os.path.exists(outdir):
        os.makedirs(outdir)


try:
    chk = DumbCheckpoint(f"{outdir}continued_solution", mode=FILE_READ)
    chk.load(s.solution, name="State")
    print("Found continued solution.")
except: # noqa
    writer = NavierStokesWriter(outdir + "continuation/", s, comm)
    s.solve_by_continuation(post_solve_cb=lambda nu: writer.write(), steps=16)
    writer.fine_write()
    chk = DumbCheckpoint(f"{outdir}continued_solution", mode=FILE_CREATE)
    chk.store(s.solution, name="State")


outu = File(outdir + "u.pvd")


def cb():
    outu.write(s.solution.split()[0])


Js2 = fsz.MoYoSpectralConstraint(1.e2, Constant(0.8), Q)
Je = EnergyRecovery(s, Q, scale=1e0, cb=cb, deformation_check=Js2)
Jr = fs.ReducedObjective(Je, s)
Js = fsz.MoYoSpectralConstraint(1.e1, Constant(0.5), Q)
J = Jr + Js
q = fs.ControlVector(Q, inner)
J.update(q, None, 1)
g = q.clone()
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
        'Gradient Tolerance': 1e-6,
        'Step Tolerance': 1e-10,
        'Iteration Limit': 150}}

params = ROL.ParameterList(params_dict, "Parameters")
problem = ROL.OptimizationProblem(J, q)
solver = ROL.OptimizationSolver(problem, params)
solver.solve()
