import firedrake as fd
import numpy as np
from firedrake.petsc import PETSc


class InnerProduct(object):

    """
    Generic implementation of an inner product.
    """

    def eval(self, u, v):
        """Evaluate inner product in primal space."""
        raise NotImplementedError

    def riesz_map(self, v, out):  # dual to primal
        """
        Compute Riesz representative of v and save it in out.

        Input:
        v: ControlVector, in the dual space
        out: ControlVector, in the primal space
        """
        raise NotImplementedError


class UflInnerProduct(InnerProduct):

    """
    Implementation of an inner product that is build on a
    firedrake.FunctionSpace.  If the ControlSpace is not itselt the
    firedrake.FunctionSpace, then an interpolation matrix between the two is
    necessary.
    """
    def __init__(self, Q, fixed_bids=[], extra_bcs=[], direct_solve=False):
        if isinstance(extra_bcs, fd.DirichletBC):
            extra_bcs = [extra_bcs]

        self.direct_solve = direct_solve
        self.fixed_bids = fixed_bids  # fixed parts of bdry
        self.params = self.get_params()  # solver parameters
        self.Q = Q

        """
        V: type fd.FunctionSpace
        I: type PETSc.Mat, interpolation matrix between V and  ControlSpace
        """
        (V, I_interp) = Q.get_space_for_inner()
        self.free_bids = list(
                           V.mesh().topology.exterior_facets.unique_markers)
        for bid in self.fixed_bids:
            self.free_bids.remove(bid)

        # Some weak forms have a nullspace. We import the nullspace if no
        # parts of the bdry are fixed (we assume that a DirichletBC is
        # sufficient to empty the nullspace).
        nsp = None
        if len(self.fixed_bids) == 0:
            nsp_functions = self.get_nullspace(V)
            if nsp_functions is not None:
                nsp = fd.VectorSpaceBasis(nsp_functions)
                nsp.orthonormalize()

        bcs = []
        # impose homogeneous Dirichlet bcs on bdry parts that are fixed.
        if len(self.fixed_bids) > 0:
            dim = V.value_size
            if dim == 2:
                zerovector = fd.Constant((0, 0))
            elif dim == 3:
                zerovector = fd.Constant((0, 0, 0))
            else:
                raise NotImplementedError
            bcs.append(fd.DirichletBC(V, zerovector, self.fixed_bids))

        if len(extra_bcs) > 0:
            bcs += extra_bcs

        if len(bcs) == 0:
            bcs = None

        a = self.get_weak_form(V)
        A = fd.assemble(a, mat_type='aij', bcs=bcs)
        ls = fd.LinearSolver(A, solver_parameters=self.params,
                             nullspace=nsp, transpose_nullspace=nsp)
        self.ls = ls
        self.A = A.petscmat
        self.interpolated = False

        # If the matrix I is passed, replace A with transpose(I)*A*I
        # and set up a ksp solver for self.riesz_map
        if I_interp is not None:
            self.interpolated = True
            ITAI = self.A.PtAP(I_interp)
            from firedrake.petsc import PETSc
            import numpy as np
            zero_rows = []

            global_num_bsplines = ITAI.size[0]
            comm = V.comm
            local_size_if_perfectly_divisible = global_num_bsplines//comm.size
            index_offset = comm.rank * local_size_if_perfectly_divisible + min(comm.rank, global_num_bsplines % comm.size)
            # if there are zero-rows, replace them with rows that
            # have 1 on the diagonal entry
            for row in range(ITAI.sizes[0][0]):
                row = row + index_offset
                (cols, vals) = ITAI.getRow(row)
                valnorm = np.linalg.norm(vals)
                if valnorm < 1e-13:
                    zero_rows.append(row)
            for row in zero_rows:
                ITAI.setValue(row, row, 1.0)
            ITAI.assemble()

            # overwrite the self.A created by get_impl
            self.A = ITAI

            # create ksp solver for self.riesz_map
            Aksp = PETSc.KSP().create(comm=V.comm)
            Aksp.setOperators(ITAI)
            Aksp.setType("preonly")
            Aksp.pc.setType("cholesky")
            Aksp.pc.setFactorSolverType("mumps")
            Aksp.setFromOptions()
            Aksp.setUp()
            self.Aksp = Aksp

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

    def get_weak_form(self, V):
        """ Weak formulation of inner product (in UFL)."""
        raise NotImplementedError

    def get_nullspace(self, V):
        """Nullspace of weak formulation of inner product (in UFL)."""
        raise NotImplementedError

    def eval(self, u, v):
        """Evaluate inner product in primal space."""
        A_u = self.A.createVecLeft()
        uvec = u.vec_ro()
        vvec = v.vec_ro()
        self.A.mult(uvec, A_u)
        return vvec.dot(A_u)

    def riesz_map(self, v, out):  # dual to primal
        """
        Compute Riesz representative of v and save it in out.

        Input:
        v: ControlVector, in the dual space
        out: ControlVector, in the primal space
        """
        if self.interpolated:
            self.Aksp.solve(v.vec_ro(), out.vec_wo())
        else:
            self.ls.solve(out.fun, v.fun)


class H1InnerProduct(UflInnerProduct):
    """Inner product on H1. It involves stiffness and mass matrices."""
    def get_weak_form(self, V):
        u = fd.TrialFunction(V)
        v = fd.TestFunction(V)
        a = fd.inner(fd.grad(u), fd.grad(v)) * fd.dx \
            + fd.inner(u, v) * fd.dx
        return a

    def get_nullspace(self, V):
        return None


class LaplaceInnerProduct(UflInnerProduct):
    """Inner product on H10. It comprises only the stiffness matrix."""
    def get_weak_form(self, V):
        u = fd.TrialFunction(V)
        v = fd.TestFunction(V)
        return fd.inner(fd.grad(u), fd.grad(v)) * fd.dx

    def get_nullspace(self, V):
        """This nullspace contains constant functions."""
        dim = V.value_size
        if dim == 2:
            n1 = fd.Function(V).interpolate(fd.Constant((1.0, 0.0)))
            n2 = fd.Function(V).interpolate(fd.Constant((0.0, 1.0)))
            res = [n1, n2]
        elif dim == 3:
            n1 = fd.Function(V).interpolate(fd.Constant((1.0, 0.0, 0.0)))
            n2 = fd.Function(V).interpolate(fd.Constant((0.0, 1.0, 0.0)))
            n3 = fd.Function(V).interpolate(fd.Constant((0.0, 0.0, 1.0)))
            res = [n1, n2, n3]
        else:
            raise NotImplementedError
        return res


class ElasticityInnerProduct(UflInnerProduct):
    """Inner product stemming from the linear elasticity equation."""
    def get_mu(self, V):
        W = fd.FunctionSpace(V.mesh(), "CG", 1)
        bcs = []
        if len(self.fixed_bids):
            bcs.append(fd.DirichletBC(W, 1, self.fixed_bids))
        if len(self.free_bids):
            bcs.append(fd.DirichletBC(W, 10, self.free_bids))
        if len(bcs) == 0:
            bcs = None
        u = fd.TrialFunction(W)
        v = fd.TestFunction(W)
        a = fd.inner(fd.grad(u), fd.grad(v)) * fd.dx
        b = fd.inner(fd.Constant(0.), v) * fd.dx
        mu = fd.Function(W)
        fd.solve(a == b, mu, bcs=bcs)
        return mu

    def get_weak_form(self, V):
        """
        mu is a spatially varying coefficient in the weak form
        of the elasticity equations. The idea is to make the mesh stiff near
        the boundary that is being deformed.
        """
        if self.fixed_bids is not None and len(self.fixed_bids) > 0:
            mu = self.get_mu(V)
        else:
            mu = fd.Constant(1.0)

        u = fd.TrialFunction(V)
        v = fd.TestFunction(V)
        return mu * fd.inner(fd.sym(fd.grad(u)), fd.sym(fd.grad(v))) * fd.dx

    def get_nullspace(self, V):
        """
        This nullspace contains constant functions as well as rotations.
        """
        X = fd.SpatialCoordinate(V.mesh())
        dim = V.value_size
        if dim == 2:
            n1 = fd.Function(V).interpolate(fd.Constant((1.0, 0.0)))
            n2 = fd.Function(V).interpolate(fd.Constant((0.0, 1.0)))
            n3 = fd.Function(V).interpolate(fd.as_vector([X[1], -X[0]]))
            res = [n1, n2, n3]
        elif dim == 3:
            n1 = fd.Function(V).interpolate(fd.Constant((1.0, 0.0, 0.0)))
            n2 = fd.Function(V).interpolate(fd.Constant((0.0, 1.0, 0.0)))
            n3 = fd.Function(V).interpolate(fd.Constant((0.0, 0.0, 1.0)))
            n4 = fd.Function(V).interpolate(fd.as_vector([-X[1], X[0], 0]))
            n5 = fd.Function(V).interpolate(fd.as_vector([-X[2], 0, X[0]]))
            n6 = fd.Function(V).interpolate(fd.as_vector([0, X[2], X[1]]))
            res = [n1, n2, n3, n4, n5, n6]
        else:
            raise NotImplementedError
        return res


class SurfaceInnerProduct(InnerProduct):

    def __init__(self, Q, free_bids=["on_boundary"]):
        (V, I_interp) = Q.get_space_for_inner()

        self.free_bids = free_bids

        u = fd.TrialFunction(V)
        v = fd.TestFunction(V)

        n = fd.FacetNormal(V.mesh())

        def surf_grad(u):
            return fd.sym(fd.grad(u) - fd.outer(fd.grad(u)*n, n))
        a = (fd.inner(surf_grad(u), surf_grad(v)) + fd.inner(u, v)) * fd.ds
        # petsc doesn't like matrices with zero rows
        a += 1e-10 * fd.inner(u, v) * fd.dx
        A = fd.assemble(a, mat_type="aij")
        A.force_evaluation()
        A = A.petscmat
        tdim = V.mesh().topological_dimension()

        lsize = fd.Function(V).vector().local_size()
        def get_nodes_bc(bc):
            nodes = bc.nodes
            return nodes[nodes < lsize]

        def get_nodes_bid(bid):
            return get_nodes_bc(fd.DirichletBC(V, fd.Constant(tdim * (0,)), bid))

        free_nodes = np.concatenate([get_nodes_bid(bid) for bid in self.free_bids])
        free_dofs = np.concatenate([tdim*free_nodes + i for i in range(tdim)])
        free_dofs = np.unique(np.sort(free_dofs))
        self.free_is = PETSc.IS().createGeneral(free_dofs)
        lgr, lgc = A.getLGMap()
        self.global_free_is_row = lgr.applyIS(self.free_is)
        self.global_free_is_col = lgc.applyIS(self.free_is)
        A = A.createSubMatrix(self.global_free_is_row, self.global_free_is_col)
        # A.view()
        A.assemble()
        self.A = A
        Aksp = PETSc.KSP().create()
        Aksp.setOperators(self.A)
        Aksp.setOptionsPrefix("A_")
        opts = PETSc.Options()
        opts["A_ksp_type"] = "cg"
        opts["A_pc_type"] = "hypre"
        opts["A_ksp_atol"] = 1e-10
        opts["A_ksp_rtol"] = 1e-10
        Aksp.setUp()
        Aksp.setFromOptions()
        self.Aksp = Aksp

    def eval(self, u, v):
        usub = u.vec_ro().getSubVector(self.global_free_is_col)
        vsub = v.vec_ro().getSubVector(self.global_free_is_col)
        A_u = self.A.createVecLeft()
        self.A.mult(usub, A_u)
        return vsub.dot(A_u)

    def riesz_map(self, v, out):  # dual to primal
        vsub = v.vec_ro().getSubVector(self.global_free_is_col)
        res = self.A.createVecLeft()
        self.Aksp.solve(vsub, res)
        outvec = out.vec_wo()
        outvec *= 0.
        outvec.setValues(self.global_free_is_col.array, res.array)
        outvec.assemble()
