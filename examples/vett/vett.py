from diffusor_mesh import create_diffusor
from firedrake import File, SpatialCoordinate, conditional, ge, \
    sinh, as_vector, DumbCheckpoint, FILE_READ, FILE_CREATE, \
    Constant, MeshHierarchy, assemble, ds, info, info_red
from rans_mixing_length import RANSMixingLengthSolver
from vett_objectives import EnergyRecovery, PressureRecovery, DissipatedEnergy
from vett_helpers import get_extra_bc, NavierStokesWriter, export_boundary
import fireshape as fs
import fireshape.zoo as fsz
import ROL
import os
import argparse
import shutil
import csv
import numpy as np
from mpi4py import MPI
comm = MPI.COMM_WORLD

parser = argparse.ArgumentParser()
parser.add_argument("--x1", type=float, default=10)
parser.add_argument("--x2", type=float, default=20)
parser.add_argument("--f", type=int, default=1)
parser.add_argument("--geo", type=int, default=1)
parser.add_argument("--inflow", type=int, default=1)
parser.add_argument("--box", type=int, default=0)
args = parser.parse_args()
x1 = args.x1
x2 = args.x2
functional = args.f
geometry = args.geo
inflow_type = args.inflow
box = args.box
label = f"x1_{x1}_x2_{x2}_geometry_{geometry}"
info_red(f"label={label}")
num_ref = 0
wakes = [round(0.05 * i, 3) for i in reversed(range(21))]
""" Create geometry """
if geometry == 1:
    clscale = 0.3 * (2**num_ref)
    h1 = 1.0
    h2 = 1.5
    xvals = [0.0, 1.0, x1-0.05, x1+0.05, x2-0.05, x2+0.05, 29, 30]
    hvals = [h1, h1, h1, h1, h2, h2, h2, h2]
    omega_free_start = 1
    omega_free_end = xvals[-2]
    Re = 1e6
    top_scale = 0.07
    mesh_code = create_rounded_diffusor(xvals, hvals, top_scale=top_scale)
else:
    h1 = 1.0
    h2 = 2./3.
    clscale = 0.2 * 2**num_ref
    xvals = [0.0, 2.0, x1-0.05, x1+0.05, x2-0.10, x2+0.10, 9.0, 15.]
    hvals = [h1, h1, h1, h1, h2, h2, h2,  h2, h2]
    omega_free_start = xvals[1]
    omega_free_end = xvals[-2]
    Re = 1e6
    top_scale = 0.07
    mesh_code = create_rounded_diffusor(xvals, hvals, top_scale=top_scale)
    wakes = [w for w in wakes if w <= 0.40+1e-7]
    wakes = [0.0]

mesh = fs.mesh_from_gmsh_code(mesh_code, clscale=clscale, smooth=100, name=label, delete_files=True)


inflow_bids = [1]
noslip_fixed_bids = [2]
noslip_free_bids = [3]
outflow_bids = [4]
symmetry_bids = [5]

control = "fem"
if control == "bsplines":
    Q = fs.BsplineBoundaryControlSpace(mesh, [(omega_free_start, omega_free_end), (0, 2)], orders=[4, 3], levels=[5, 1], fixed_dims=[0], boundary_regularities=[2, 1])
else:
    # Q = fs.FeMultiGridBoundaryControlSpace(mesh, refinements=num_ref, order=1)
    Q = fs.FeControlSpace(mesh)

mesh_m = Q.mesh_m

local_mesh_size = np.asarray([mesh_m.num_cells()])
global_mesh_size = np.asarray([0])
comm.Reduce(local_mesh_size, global_mesh_size)
comm.Bcast(global_mesh_size)
global_mesh_size = global_mesh_size[0]
info("Number of cells: ", global_mesh_size)


(V_control, _) = Q.get_space_for_inner()
if control == "bsplines":
    inner = fs.LaplaceInnerProduct(Q, fixed_bids=[1, 2, 3, 4])
else:
    extra_bc = get_extra_bc(V_control, omega_free_start, omega_free_end)
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

def get_inflow(wake):
    if inflow_type == 1:
        xvel = linear_wake_function(y, wake)
    else:
        xvel = sinh_wake_function(2.0 * y, -wake, 2)
        integral = sum([assemble(xvel * ds(bid, domain=mesh_m)) for bid in inflow_bids])
        xvel = xvel/integral
    return as_vector([smoother * xvel, 0])


inflow_expr = get_inflow(wakes[0])
solver_t = RANSMixingLengthSolver
pvel = 3
s = solver_t(mesh_m, inflow_bids=inflow_bids,
             inflow_expr=inflow_expr,
             noslip_bids=noslip_free_bids + noslip_fixed_bids,
             symmetry_bids=symmetry_bids,
             nu=1.0/Re, velocity_degree=pvel)

(u, p) = s.solution.split()
(v, q) = s.solution_adj.split()

def change_bc(wake):
    new_inflow = get_inflow(wake)
    s.problem.bcs[0]._original_val = new_inflow
    s.problem.bcs[0].function_arg = new_inflow
    s.problem.bcs[0]._original_val = s.problem.bcs[0].function_arg

outdir = f"output11/{label}/{global_mesh_size}/"

if comm.rank == 0:
    if not os.path.exists(outdir):
        os.makedirs(outdir)
comm.Barrier()

try:
    dumpfile = f"{outdir}continued_solution"
    dumpfile = comm.bcast(dumpfile, root=0)
    with DumbCheckpoint(dumpfile, mode=FILE_READ) as chk:
        chk.load(s.solution, name="State")
    print("Found continued solution.")
except: # noqa
    writer = NavierStokesWriter(outdir + "continuation/", s, comm)
    s.solve_by_continuation(post_solve_cb=lambda nu: writer.write(), steps=21)
    with DumbCheckpoint(f"{outdir}continued_solution", mode=FILE_CREATE) as chk:
        chk.store(s.solution, name="State")

export_boundary(mesh_m, outdir)
if geometry == 1:
    Js2 = None
else:
    Js2 = fsz.MoYoSpectralConstraint(1.e2, Constant(0.8), Q)

""" Start setting up the optimization """
if functional == 1:
    f = PressureRecovery(s, Q, scale=1e-2, deformation_check=Js2)
    c2 = f.scale * 1e1
elif functional == 2:
    f = EnergyRecovery(s, Q, scale=1e0, deformation_check=Js2, quadrature_degree=100)
    c2 = f.scale * 1e1
elif functional == 3:
    f = DissipatedEnergy(s, Q, scale=1e-2, deformation_check=Js2)
    c2 = f.scale * 1e1
else:
    raise NotImplementedError

if geometry == 1:
    upper_bnd = Q.T.copy(deepcopy=True).interpolate(Constant((+100, 1.5)))
    lower_bnd = Q.T.copy(deepcopy=True).interpolate(Constant((-100, 1.0)))
else:
    upper_bnd = Q.T.copy(deepcopy=True).interpolate(Constant((+100, 1.)))
    lower_bnd = Q.T.copy(deepcopy=True).interpolate(Constant((-100, h2)))

if box == 1:
    c1 = 1e-2 * f.scale
else:
    c1 = 1e-10

Jr = fs.ReducedObjective(f, s)
Js = fsz.MoYoSpectralConstraint(c2, Constant(0.5), Q)
if geometry == 1:
    Js = 1e-10 * Js
Jb = fsz.MoYoBoxConstraint(c1, noslip_free_bids, Q, lower_bound=lower_bnd,
                           upper_bound=upper_bnd)
J = Jr + Js + Jb
# J = 1e-1 * J
q = fs.ControlVector(Q, inner)
J.update(q, None, 1)
g = q.clone()
J.gradient(g, q, None)
gradtol = (1e-4) * g.dot(g)**0.5
g.scale(1e2)
J.checkGradient(q, g, 6, 1)
import sys; sys.exit(1)
params_dict = {
    'General': {
        'Secant': {'Type': 'Limited-Memory BFGS',
                   'Maximum Storage': 10}},
    'Step': {
        'Type': 'Line Search',
        'Line Search': {'Descent Method': {
            'Type': 'Quasi-Newton Step'}}},
    'Status Test': {
        'Gradient Tolerance': gradtol,
        'Step Tolerance': 1e-10,
        'Iteration Limit': 150}}

functional_outdir = outdir + f.__class__.__name__ + f"_box_{box}/"
for wake in wakes:
    change_bc(wake)
    s.solve()
    wake_outdir = functional_outdir + f"inflow_{inflow_type}_wake_{wake:.3f}/"

    if box == 1:
        itercounts = [30] * 5
    else:
        if wake == wakes[0]:
            itercounts = [30]
        else:
            itercounts = [20]

    for i in range(len(itercounts)):
        maxiter = itercounts[i]
        iter_outdir = wake_outdir + f"{i}/"
        # try:
            # with DumbCheckpoint(f"{iter_outdir}q", mode=FILE_READ) as chk:
            #     chk.load(q.fun, name="Control")
            # with DumbCheckpoint(f"{iter_outdir}u", mode=FILE_READ) as chk:
            #     chk.load(s.solution, name="State")
            # print(f"Found deformed domain after optimization iteration {i}.")
        # except: # noqa
        if comm.rank == 0:
            if os.path.exists(iter_outdir):
                shutil.rmtree(iter_outdir)
            os.makedirs(iter_outdir)
            csvfile = open(f"{iter_outdir}log.csv", "w")
            csvwriter = csv.writer(csvfile, delimiter=",")
            csvwriter.writerow(["functional"])
        comm.Barrier()
        writer = NavierStokesWriter(iter_outdir, s, comm)

        def optim_cb():
            writer.coarse_write()
            val = f.val()
            if comm.rank == 0:
                csvwriter.writerow([val])  # , myp1.violation()])
                csvfile.flush()
        J.cb = optim_cb

        params_dict["Status Test"]["Iteration Limit"] = maxiter
        params = ROL.ParameterList(params_dict, "Parameters")
        problem = ROL.OptimizationProblem(J, q)
        solver = ROL.OptimizationSolver(problem, params)
        solver.solve()
        writer.fine_write()
        if comm.rank == 0:
            csvfile.close()
        export_boundary(mesh_m, iter_outdir)
        # with DumbCheckpoint(f"{iter_outdir}q", mode=FILE_CREATE) as chk:
        #     chk.store(q.fun, name="Control")
        # with DumbCheckpoint(f"{iter_outdir}u", mode=FILE_CREATE) as chk:
        #     chk.store(s.solution, name="State")
    if box == 1:
        Jb.c *= 2.

    print(f"label={label}")
