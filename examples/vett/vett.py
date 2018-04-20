from diffusor_mesh_rounded import create_rounded_diffusor
from gmsh_helpers import mesh_from_gmsh_code
from firedrake import File, Constant
from rans_mixing_length import RANSMixingLengthSolver


xvals = [0.0, 0.9, 1.0, 3.9, 4.0, 5.0]
hvals = [1.0, 1.0, 1.0, 0.7, 0.7, 0.7]
mesh_code = create_rounded_diffusor(xvals, hvals)
mesh = mesh_from_gmsh_code(mesh_code)

inflow_bids = [1]
noslip_fixed_bids = [2]
noslip_free_bids = [3]
outflow_bids = [4]
symmetry_bids = [5]

inflow_expr = Constant((1.0, 0.0))
solver_t = RANSMixingLengthSolver
Re = 1e3
pvel = 3
s = solver_t(mesh, inflow_bids=inflow_bids,
             inflow_expr=inflow_expr,
             noslip_bids=noslip_free_bids + noslip_fixed_bids,
             symmetry_bids=symmetry_bids,
             nu=1.0/Re, velocity_degree=pvel)

s.solve()
File("u.pvd").write(s.solution.split()[0])
