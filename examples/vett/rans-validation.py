from diffusor_mesh import create_diffusor
from firedrake import File, SpatialCoordinate, conditional, ge, \
    sinh, as_vector, DumbCheckpoint, FILE_READ, FILE_CREATE, \
    Constant, MeshHierarchy, assemble, ds, info, info_red, File, FunctionSpace, Function, div, \
    FacetNormal, dx, assemble, inner
from rans_mixing_length import RANSMixingLengthSolver
from vett_objectives import EnergyRecovery, PressureRecovery, DissipatedEnergy
import fireshape as fs
import fireshape.zoo as fsz
import ROL
import os
import argparse
import shutil
import csv
import numpy as np
import argparse
from mpi4py import MPI
comm = MPI.COMM_WORLD

parser = argparse.ArgumentParser()
parser.add_argument("--x1", type=int, default=5)
parser.add_argument("--x2", type=int, default=19)
args = parser.parse_args()
x1 = args.x1
x2 = args.x2
# x1 = 5.
# x2 = 19.
h1 = 1.0
h2 = 1.5
functional = 1
inflow_type = 1
wokeness = 1.
xvals = [0., 0.5, x1, x2, 29.5, 30]
hvals = [h1, h1, h1, h2, h2, h2]
clscale = 0.5
Re = 1e6
top_scale = 0.05
mesh_code = create_diffusor(xvals, hvals, top_scale=top_scale, rounded=False)
filename = "%i-%i" % (x1, x2)
mesh = fs.mesh_from_gmsh_code(mesh_code, clscale=clscale, smooth=100, delete_files=True, name=filename)
Q = fs.FeControlSpace(mesh)
mesh = Q.mesh_m

inflow_bids = [1]
noslip_fixed_bids = [2]
noslip_free_bids = [3]
outflow_bids = [4]
symmetry_bids = [5]

x, y = SpatialCoordinate(mesh)
eps = 0.002
smoother = conditional(ge(y, 1-eps), (1-((1/eps)*(y-(1-eps)))**4)**4, 1.0)
smoother = 1

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
        integral = sum([assemble(xvel * ds(bid, domain=mesh)) for bid in inflow_bids])
        xvel = xvel/integral
    return as_vector([smoother * xvel, 0])


inflow_expr = get_inflow(wokeness)
solver_t = RANSMixingLengthSolver
pvel = 3
s = solver_t(mesh, inflow_bids=inflow_bids,
             inflow_expr=inflow_expr,
             noslip_bids=noslip_free_bids + noslip_fixed_bids,
             symmetry_bids=symmetry_bids,
             nu=1.0/Re, velocity_degree=pvel)

(u, p) = s.solution.split()
(v, q) = s.solution_adj.split()

s.solve_by_continuation(steps=21)#, post_solve_cb= lambda mu: print("Done solving for mu=%f" % mu))

if functional == 1:
    f = PressureRecovery(s, Q)
    c2 = f.scale * 1e1
elif functional == 2:
    f = EnergyRecovery(s, Q)
    c2 = f.scale * 1e1
elif functional == 3:
    f = DissipatedEnergy(s, Q)
    c2 = f.scale * 1e1
else:
    raise NotImplementedError
outdir = "validation"
if not os.path.exists(outdir):
    os.makedirs(outdir)
with open(outdir + "/%i-%i.csv" % (x1, x2), "w") as fi:
    fi.write("%i" % x1)
    fi.write("%i" % x2)
    fi.write("%f" % f.value(None, None))

print("--------------------")
print("dofs:", s.V.dim())
print("f", f.value(None, None))
print("vol form 1", assemble(div(u*p) * dx(6)))
print("vol form 2", assemble(div(u*p) * dx(7)))
n = FacetNormal(mesh)
print("surface form", assemble(inner(u,n)*p*ds))
print("--------------------")
File(outdir + "/%s-u.pvd" % filename).write(u)
File(outdir + "/%s-p.pvd" % filename).write(p)
# DG = FunctionSpace(mesh, "DG", 0)
# File("integrand2.pvd").write(Function(DG).interpolate(div(u*p)))
