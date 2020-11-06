#!/usr/bin/env python
# -*- coding: utf-8 -*-

from firedrake import *
from firedrake_adjoint import *
mesh = TorusMesh(100,40,10,3)
x = SpatialCoordinate(mesh)
W = VectorFunctionSpace(mesh, "CG", 1, dim=3)
xc = Function(W).interpolate(x)

n = cross(xc.dx(0),xc.dx(1))*conditional(ge(x[2],0), 1, -1)
n = n/sqrt(inner(n,n))
mesh.init_cell_orientations(n)

V = FunctionSpace(mesh, "CG", 1)

# Solution at the current time level
u = Function(V)

# Solution at the previous time level
u_old = Function(V)

# Test function
v = TestFunction(V)

# Initial condition
g = Function(V) #interpolate(Expression("sin(x[2])*cos(x[1])", degree=2), V)

# Body force
bf = Function(V).interpolate(conditional(gt(x[1],0), 10,0)*conditional(le(abs(x[0]),2),1,0))
 
# Thermal diffusivity
nu = 10.0


# Set the options for the time discretization
T = 1.
t = 0.0
step = 0.1

# Define the variational formulation of the problem
F = u * v * dx - u_old * v * dx + step * nu * inner(grad(v), grad(u)) * dx -step*inner(bf,v)*dx

# The next step is to solve the time-dependent forward problem.

u_pvd = File("output/u.pvd")

# Execute the time loop
u_old.assign(g)

pert = Function(W)
mesh.coordinates.assign(mesh.coordinates+pert)
J=0
while t <= T:
    t += step

    solve(F == 0, u)
    u_old.assign(u)

    u_pvd.write(u)
    
    J += assemble(inner(bf,u)*dx)

control = Control(pert)
Jhat = ReducedFunctional(J, control)
dJctrl = Jhat.derivative()
temp = Function(W, name = "ShapeGrad")
temp.assign(dJctrl)
normals = Function(W,name="normals").interpolate(n)
File("output/deriv.pvd").write(temp,normals)



