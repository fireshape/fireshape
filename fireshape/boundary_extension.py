import firedrake as fd
from firedrake.petsc import PETSc
from mpi4py import MPI
import numpy as np


class ElasticityExtension(object):

    def __init__(self, V, fixed_dims=[], direct_solve=False):
        if isinstance(fixed_dims, int):
            fixed_dims = [fixed_dims]
        self.V = V
        self.fixed_dims = fixed_dims
        self.direct_solve = direct_solve
        self.zero = fd.Constant(V.mesh().topological_dimension() * (0,))
        u = fd.TrialFunction(V)
        v = fd.TestFunction(V)
        self.zero_fun = fd.Function(V)
        self.a = 1e-2 * fd.inner(u, v) * fd.dx + fd.inner(fd.sym(fd.grad(u)), fd.sym(fd.grad(v))) * fd.dx
        self.bc_fun = fd.Function(V)
        if len(self.fixed_dims) == 0:
            bcs = [fd.DirichletBC(self.V, self.bc_fun, "on_boundary")]
        else:
            bcs = []
            for i in range(self.V.mesh().topological_dimension()):
                if i in self.fixed_dims:
                    bcs.append(fd.DirichletBC(self.V.sub(i), 0, "on_boundary"))
                else:
                    bcs.append(fd.DirichletBC(self.V.sub(i), self.bc_fun.sub(i), "on_boundary"))
        self.A_ext= fd.assemble(self.a, bcs=bcs, mat_type="aij")
        self.ls_ext= fd.LinearSolver(self.A_ext, solver_parameters=self.get_params())
        self.A_adj = fd.assemble(self.a, bcs=fd.DirichletBC(self.V, self.zero, "on_boundary"), mat_type="aij")
        self.ls_adj =  fd.LinearSolver(self.A_adj, \
                                       solver_parameters=self.get_params())

    def extend(self, bc_val, out):
        self.bc_fun.assign(bc_val)
        self.ls_ext.solve(out, self.zero_fun)

    def solve_homogeneous_adjoint(self, rhs, out):
        for i in self.fixed_dims:
            temp = rhs.sub(i)
            temp *= 0
        self.ls_adj.solve(out, rhs)

    def apply_adjoint_action(self, x, out):
        # fd.assemble(fd.action(self.a, x), tensor=out)
        out.assign(fd.assemble(fd.action(self.a, x)))

    def get_params(self):
        """PETSc parameters to solve linear system."""
        params = {
            'ksp_rtol': 1e-11,
            'ksp_atol': 1e-11,
            'ksp_stol': 1e-16,
            'ksp_type': 'cg',
        }
        if self.direct_solve:
            params["pc_type"] = "cholesky"
            params["pc_factor_mat_solver_type"] = "mumps"
        else:
            params["pc_type"] = "hypre"
            params["pc_hypre_type"] = "boomeramg"
        return params


class NormalExtension(object):

    def __init__(self, V, allow_tangential=True):
        self.allow_tangential = allow_tangential
        self.V = V
        mesh = V.mesh()
        tdim = mesh.topological_dimension() 
        self.zero = fd.Constant(tdim * (0,))
        u = fd.TrialFunction(V)
        v = fd.TestFunction(V)

        if allow_tangential:
            V_mult = fd.FunctionSpace(mesh, "CG", 1)
        else:
            V_mult = fd.VectorFunctionSpace(mesh, "CG", 1)

        a = 1e-4 * fd.inner(u, v) * fd.dx + fd.inner(fd.sym(fd.grad(u)), fd.sym(fd.grad(v))) * fd.dx
        if tdim == 2:
            def B(u):
                return fd.as_vector([fd.grad(u[0])[0]-fd.grad(u[1])[1], fd.grad(u[0])[1]+fd.grad(u[1])[0]])
            a += fd.inner(B(u), B(v)) * fd.dx
        A = fd.assemble(a, mat_type="aij")
        A.force_evaluation()
        A = A.petscmat

        lam = fd.TestFunction(V_mult)
        w = fd.TrialFunction(V)
        n = fd.FacetNormal(mesh)
        if allow_tangential:
            B = fd.assemble(fd.inner(w, n * lam) * fd.ds)
        else:
            B = fd.assemble(fd.inner(w, lam) * fd.ds)

        B.force_evaluation()
        B = B.petscmat

        boundary_nodes = V.boundary_nodes("on_boundary", "topological")
        self.boundary_nodes = boundary_nodes
        num_bdry_nodes = len(boundary_nodes)
        if allow_tangential:
            row_is = PETSc.IS().createGeneral(boundary_nodes)
        else:
            boundary_dofs = np.concatenate([tdim*boundary_nodes + i for i in range(tdim)])
            boundary_dofs = np.unique(np.sort(boundary_dofs))
            self.boundary_dofs = boundary_dofs
            row_is = PETSc.IS().createGeneral(boundary_dofs)

        B = B.createSubMatrix(row_is)
        BT = PETSc.Mat()
        BT.createTranspose(B)

        comm = MPI.COMM_WORLD
        rank = comm.rank
        size = comm.size
        if size>1:
            raise NotImplementedError
        M = PETSc.Mat().create(comm=comm)

        M.setType(PETSc.Mat.Type.AIJ)
        Msize = A.size[0] + B.size[0]
        M.setSizes((Msize, Msize))
        M.setUp()
        for row in range(A.size[0]):
            (cols, vals) = A.getRow(row)
            M.setValues([row], cols, vals)

        for row in range(B.size[0]):
            (cols, vals) = B.getRow(row)
            M.setValues([row+A.size[0]], cols, vals)
            M.setValues(cols, [row+A.size[0]], vals)
            M.setValue(row+A.size[0], row+A.size[0], 0.0)


        M.assemble()
        print(M.size)


        Mksp = PETSc.KSP().create()
        Mksp.setOperators(M)
        Mksp.setOptionsPrefix("M_")
        opts = PETSc.Options()
        opts["M_ksp_monitor"] = None
        opts["M_ksp_maxit"] = 100
        opts["M_ksp_type"] = "preonly"
        opts["M_pc_type"] = "cholesky"
        Mksp.setUp()
        Mksp.setFromOptions()

        self.M = M
        self.Mksp = Mksp
        self.A = A
        self.B = B
        self.mesh = mesh
        self.V_mult = V_mult
        self.tdim = tdim

    def extend(self, theta, out):
        x = self.M.createVecRight()
        b = self.M.createVecRight()
        num_dofs = self.A.size[0]
        v = fd.TestFunction(self.V_mult)
        n = fd.FacetNormal(self.mesh)

        if self.allow_tangential:
            rhs = fd.assemble(fd.inner(v, theta) * fd.ds).vector().get_local()[self.boundary_nodes]
        else:
            rhs = fd.assemble(fd.inner(v, n * theta) * fd.ds).vector().get_local()[self.boundary_dofs]

        for i in range(len(rhs)):
            b[num_dofs+i] = rhs[i]

        self.Mksp.solve(b, x)
        resvec = out.vector()
        resvec.set_local(x[0:num_dofs])

    def adjoint(self, residual, out):
        num_dofs = self.A.size[0]
        x = self.M.createVecRight()
        b = self.M.createVecRight()

        b[0:num_dofs] = residual.vector().get_local()
        self.Mksp.solve(b, x)
        if self.allow_tangential:
            out.vector()[self.boundary_nodes] = x[num_dofs:]
            out.assign(fd.assemble(out * fd.TestFunction(self.V_mult) * fd.ds))
        else:
            temp = fd.Function(self.V_mult)
            for i in range(self.tdim):
                temp.vector()[self.boundary_nodes, i] = x[num_dofs+i::self.tdim]
            n = fd.FacetNormal(self.mesh)
            out.assign(fd.assemble(fd.inner(temp, n*fd.TestFunction(out.function_space())) * fd.ds))
            # raise NotImplementedError
