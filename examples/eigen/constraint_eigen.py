import firedrake as fd
import firedrake_adjoint as fda
import fireshape as fs
from slepc4py import SLEPc
from firedrake.petsc import PETSc


def EigenValueSolver(V, bcs):
    v = fd.TestFunction(V)
    du = fd.TrialFunction(V)

    # Setup eigenvalue problem
    k = fd.inner(fd.grad(du), fd.grad(v))*fd.dx
    m = fd.inner(du, v)*fd.dx

    # Assemble stiffness and mass matrix
    K = fd.assemble(k, bcs=bcs)
    M = fd.assemble(m, bcs=bcs)

    from firedrake.preconditioners.patch import bcdofs
    lgmap = V.dof_dset.lgmap
    for bc in bcs:
        # Ensure symmetry of M
        M.M.handle.zeroRowsColumns(lgmap.apply(bcdofs(bc)), diag=0)

    # Create the SLEPc eigensolver
    e_sp = {"eps_type": "krylovschur",
            "eps_target": -1,
            # "eps_monitor_all": None,
            # "eps_converged_reason": None,
            "eps_nev": 5,
            "st_type": "sinvert",
            "st_ksp_type": "preonly",
            "st_pc_type": "lu",
            "st_pc_factor_mat_solver_type": "mumps"}

    # Apply Options
    opts = PETSc.Options()
    for k in e_sp:
        opts[k] = e_sp[k]
    eps = SLEPc.EPS().create(comm=fd.COMM_WORLD)
    eps.setOperators(K.M.handle, M.M.handle)
    eps.setWhichEigenpairs(eps.Which.SMALLEST_MAGNITUDE)
    eps.setProblemType(eps.ProblemType.GHEP)
    eps.setFromOptions()

    eps.solve()

    eigenvalues = []
    eigenfunctions = []
    eigenfunction = fd.Function(V, name="Eigenfunction")

    for i in range(eps.getConverged()):
        lmbda = eps.getEigenvalue(i)
        assert lmbda.imag == 0
        eigenvalues.append(lmbda.real)
        with eigenfunction.dat.vec_wo as x:
            eps.getEigenvector(i, x)
        eigenfunctions.append(eigenfunction.copy(deepcopy=True))

    # outfile = fd.File("output/eigen.pvd")
    # for e in eigenfunctions:
    #     outfile.write(e)

    return eigenvalues, eigenfunctions


class EigenObjective(fs.PDEconstrainedObjective):
    """
    Find the smallest N eigenvalues
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Functional
        self.J = 0
        mesh = self.Q.mesh_m
        self.V = fd.FunctionSpace(mesh, "CG", 1)      # Solution Space
        self.bcs = [fd.DirichletBC(self.V, fd.Constant(0), "on_boundary")]

        # Eigenvalue Space
        self.R = fd.FunctionSpace(mesh, "R", 0)
        self.W = self.V*self.R
        z = fd.Function(self.W)
        zt = fd.TestFunction(self.W)

        # Split eigenfunction and eigenvalue space
        zf, ze = fd.split(z)
        zft, zet = fd.split(zt)
        self.R = fd.inner(fd.grad(zf), fd.grad(zft))*fd.dx \
            - ze*fd.inner(zf, zft)*fd.dx
        # Add Constraint Equation
        self.R += zet*(fd.inner(zf, zf)-fd.Constant(1.0))*fd.dx

        self.bcs_eig = fd.DirichletBC(self.W.sub(0),
                                      fd.Constant(0), "on_boundary")
        self.z = z
        self.zt = zt

        sp = {"mat_type": "matfree",
              "ksp_type": "gmres",
              "pc_type": "fieldsplit",
              "pc_fieldsplit_type": "schur",
              "pc_fieldsplit_schur_fact_type": "full",
              "pc_fieldsplit_0_fields": "0",
              "pc_fieldsplit_1_fields": "1",
              "fieldsplit_0_ksp_type": "preonly",
              "fieldsplit_0_pc_type": "python",
              "fieldsplit_0_pc_python_type": "firedrake.AssembledPC",
              "fieldsplit_0_assembled_pc_type": "lu",
              "fieldsplit_0_assembled_pc_factor_mat_solver_type": "mumps",
              "fieldsplit_1_ksp_type": "gmres",
              # "fieldsplit_1_ksp_monitor_true_residual": None,
              "fieldsplit_1_ksp_max_it": 1,
              "fieldsplit_1_ksp_convergence_test": "skip",
              "fieldsplit_1_pc_type": "none"}
        self.sp = sp
        self.outfile = fd.File("output/shape.pvd")

    def solvePDE(self):
        self.J = 0
        # FIXME: Slack conversation has previously warned about using
        # pause/continue
        fda.pause_annotation()
        evs, efs = EigenValueSolver(self.V, self.bcs)
        fda.continue_annotation()
        i = 0
        self.z.sub(0).assign(efs[i])
        self.z.sub(1).assign(evs[i])
        fd.solve(self.R == 0, self.z, bcs=self.bcs_eig,
                 solver_parameters=self.sp)
        evec, evl = self.z.split()
        self.J += fd.assemble((evl-fd.Constant(14.0))**2*fd.dx)
        self.outfile.write(evec)

    def objective_value(self):
        return self.J

    def cb(self):
        # FIXME: Call back is not working
        u, e = self.z.split()


if __name__ == "__main__":
    n = 20
    mesh = fd.UnitSquareMesh(n, n)
    Q = fs.FeControlSpace(mesh)
    inner = fs.LaplaceInnerProduct(Q)
    q = fs.ControlVector(Q, inner)

    # setup PDE
    J = EigenObjective(Q)
    J.solvePDE()
