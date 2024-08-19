import firedrake as fd
import fireshape as fs
import fireshape.zoo as fsz
import ROL
from PDEconstraint_pipe import NavierStokesSolver
from objective_pipe import PipeObjective

# setup problem
mesh_r = fd.Mesh("pipe.msh")
mesh_c = fd.Mesh("pipe_control.msh")

S = fd.FunctionSpace(mesh_c, "DG", 0)

I = fd.Function(S, name="indicator")
fd.par_loop(("{[i] : 0 <= i < f.dofs}", "f[i, 0] = 1.0"),
         fd.dx(2),
         {"f": (I, fd.WRITE)})

# save state variable evolution in file u2.pvd or u3.pvd
out = fd.File("solution/indicator_test.pvd")
out.write(I)