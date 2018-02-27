import firedrake as fd

class InnerProductImpl(object):
    """I suggest to move this to InnerProduct.get_impl."""
    def __init__(self, ls, A):
        self.ls = ls
        self.A = A

    def riesz_map(self, v, out):  # dual to primal
        # expects two firedrake vector objects
        if v.fun is None or out.fun is None:
            self.ls.ksp.solve(v.vec, out.vec)  # Won't do boundary conditions
        self.ls.solve(out.fun, v.fun)

    def eval(self, u, v):
        """Evaluate inner product in primal space."""
        A_u = self.A.createVecLeft()
        self.A.mult(u.vec, A_u)
        return v.vec.dot(A_u)


class InnerProduct(object):
    """
    Generic implementation of metric for ControlSpace.

    The input fixed_bids is a list of bdry parts that are not free to move.
    The method get_impl links the chosen inner product to ControlSpace.V_r.
    """
    def __init__(self, fixed_bids=[]):
        self.fixed_bids = fixed_bids
        self.params = self.get_params() # solver parameters

    def get_params(self):
        """PETSc parameters to solve linear system."""
        return {
                'ksp_solver': 'gmres',
                'pc_type': 'lu',
                'pc_factor_mat_solver_package': 'mumps',
                # 'ksp_monitor': True
                }

    def get_weak_form(self, V):
        """ Weak formulation of inner product in UFL (Firedrake)."""
        raise NotImplementedError

    def get_nullspace(self, V):
        """Nullspace of weak formulation of inner product (in UFL, Firedake)."""
        raise NotImplementedError

    def get_impl(self, V):
        """Link metric to ControlSpace.V_r."""
        self.free_bids = list(V.mesh().topology.exterior_facets.unique_markers)
        for bid in self.fixed_bids:
            self.free_bids.remove(bid)

        # what is this?
        nsp = None
        if len(self.fixed_bids) == 0:
            nsp_functions = self.get_nullspace(V)
            if nsp_functions is not None:
                nsp = fd.VectorSpaceBasis(nsp_functions)
                nsp.orthonormalize()

        # impose homogeneous Dirichlet bcs on bdry parts that are not free to move
        if len(self.fixed_bids) > 0:
            dim = V.value_size
            if dim == 2:
                zerovector = fd.Constant((0, 0))
            elif dim == 3:
                zerovector = fd.Constant((0, 0, 0))
            else:
                raise NotImplementedError
            bc = fd.DirichletBC(V, zerovector, self.fixed_bids)
        else:
            bc = None

        a = self.get_weak_form(V)
        A = fd.assemble(a, mat_type='aij', bcs=bc)
        A = fd.as_backend_type(A).mat()
        ls = fd.LinearSolver(A, solver_parameters=self.params, nullspace=nsp,
                             transpose_nullspace=nsp)
        # it would be nice if we can decide here whether to call
        # InnerProductImpl or InterpolatedInnerProduct
        # for instance, InnerProductImpl.eval could be put here
        return InnerProductImpl(ls, A)


    #this is now in InnerProductImpl, shall we remove it?
    def riesz_map(self, v, out): # dual to primal
        # expects two FEControlObjects
        if v.fun is None or out.fun is None:
            self.ls.ksp.solve(v.vec, out.vec) # Won't do boundary conditionsd
        self.ls.solve(out.fun, v.fun) #suggestion: force this

    #this is now in InnerProductImpl, shall we remove it?
    def eval(self, u, v): # inner product in primal space
        # expects two FEControlObjects
        A_u = self.A.createVecLeft()
        self.A.mult(u.vec, A_u)
        return v.vec.dot(A_u)

class H1InnerProduct(InnerProduct):
    """Inner product on H1. It involves stiffness and mass matrices."""
    def get_weak_form(self, V):
        u = fd.TrialFunction(V)
        v = fd.TestFunction(V)
        a = fd.inner(fd.grad(u), fd.grad(v)) * fd.dx \
            + fd.inner(u, v) * fd.dx
        return a


class LaplaceInnerProduct(InnerProduct):
    """Inner product on H10. It comprises only the stiffness matrix."""
    def get_weak_form(self, V):
        u = fd.TrialFunction(V)
        v = fd.TestFunction(V)
        return fd.inner(fd.grad(u), fd.grad(v)) * fd.dx

    def get_nullspace(self, V):
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


class ElasticityInnerProduct(InnerProduct):
    """Inner product stemming from the linear elasticity equation."""
    def get_mu(self, V):
        W = fd.FunctionSpace(V.mesh(), "CG", 1)
        bc_fix = fd.DirichletBC(W, 1, self.fixed_bids)
        bc_free = fd.DirichletBC(W, 10, self.free_bids)
        u = fd.TrialFunction(W)
        v = fd.TestFunction(W)
        a = fd.inner(fd.grad(u), fd.grad(v)) * fd.dx
        b = fd.inner(fd.Constant(0.), v) * fd.dx
        mu = fd.Function(W)
        fd.solve(a == b, mu, bcs = [bc_fix, bc_free])
        return mu

    def get_weak_form(self, V):
        mu = self.get_mu(V)
        u = fd.TrialFunction(V)
        v = fd.TestFunction(V)
        return mu * fd.inner(fd.sym(fd.grad(u)), fd.sym(fd.grad(v))) * fd.dx

    def get_nullspace(self, V):
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


class InterpolatedInnerProduct(InnerProduct):
    """
    Shouldn't this inherit from InnerProductImpl?

    Inner products for ControlVector with self.fun = None.

    Assemble the matrix representation of the inner product by multiplying
    the matrix representation of the original inner product with the interpolation
    matrix I.

    ISSUE: this cannot be correct if the support of the nonFEM basis vector fields is
    larger than the physical domain, or if the computational domains has holes
    that intersect the support of nonFEM basis vector field
    """
    def __init__(self, A, I):
        ITAI = A.PtAP(I)
        from firedrake.petsc import PETSc
        import numpy as np
        zero_rows = []
        for row in range(ITAI.size[0]):
            (cols, vals) = ITAI.getRow(row)
            valnorm = np.linalg.norm(vals)
            if valnorm < 1e-13:
                zero_rows.append(row)
        for row in zero_rows:
            ITAI.setValue(row, row, 1.0)
        ITAI.assemble()
        self.A = ITAI
        #create solver
        # Aksp = PETSc.KSP().create(comm=self.comm)
        Aksp = PETSc.KSP().create()
        Aksp.setOperators(ITAI)
        Aksp.setType("preonly")
        Aksp.pc.setType("cholesky")
        Aksp.pc.setFactorSolverPackage("mumps")
        Aksp.setFromOptions()
        Aksp.setUp()
        self.Aksp = Aksp

    def riesz_map(self, v, out):
        self.Aksp.solve(v.vec, out.vec)

    #this is exactly the same code of InnerProductImpl.eval
    def eval(self, u, v):
        A_u = self.A.createVecLeft()
        self.A.mult(u.vec, A_u)
        return v.vec.dot(A_u)
