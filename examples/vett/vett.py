from diffusor_mesh_rounded import create_rounded_diffusor
from gmsh_helpers import mesh_from_gmsh_code
from firedrake import File, Constant
from rans_mixing_length import RANSMixingLengthSolver
from vett_objectives import EnergyRecovery
import fireshape as fs


xvals = [0.0, 0.9, 1.0, 3.9, 4.0, 5.0]
hvals = [1.0, 1.0, 1.0, 0.7, 0.7, 0.7]
mesh_code = create_rounded_diffusor(xvals, hvals)
mesh = mesh_from_gmsh_code(mesh_code, clscale=2.0)
inflow_bids = [1]
noslip_fixed_bids = [2]
noslip_free_bids = [3]
outflow_bids = [4]
symmetry_bids = [5]
inner = fs.LaplaceInnerProduct(fixed_bids=inflow_bids + noslip_fixed_bids
                               + outflow_bids + symmetry_bids)
Q = fs.FeMultiGridControlSpace(mesh, inner, refinements=2)
mesh_m = Q.mesh_m


inflow_expr = Constant((1.0, 0.0))
solver_t = RANSMixingLengthSolver
Re = 1e3
pvel = 2
s = solver_t(mesh_m, inflow_bids=inflow_bids,
             inflow_expr=inflow_expr,
             noslip_bids=noslip_free_bids + noslip_fixed_bids,
             symmetry_bids=symmetry_bids,
             nu=1.0/Re, velocity_degree=pvel)
s.solve()
File("u.pvd").write(s.solution.split()[0])
J = EnergyRecovery(s, Q, scale=1e-2)
Jr = fs.ReducedObjective(J, s)

q = fs.ControlVector(Q)
Jr.update(q, None, 1)
g = fs.ControlVector(Q)
Jr.gradient(g, q, None)
Jr.checkGradient(q, g, 7, 1)
