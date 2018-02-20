import firedrake as fd

class InnerProduct(object):
    "Choose the metric for Riesz representatives of Frechet derivatives.
    To compute Riesz representatives, exploit Firedrake capabilities

    The input fixed_bids is a list of bdry parts that are not free to move
    "
    def __init__(self, mesh, fixed_bids=[]):

        element = mesh.coordinates.function_space().ufl_element()
        self.V = fd.FunctionSpace(mesh, element) #shall we call this self.V_r?
        self.dim = mesh.topological_dimension() #dimension of physical space

        #assemble Riesz matrix A
        a = self.get_weak_form()
        A = fd.assemble(a, mat_type='aij')

        #incorporate homogenous Dirichlet bcs in Riesz matrix A
        if len(fixed_bids) > 0:
            if self.dim <= 3:
                zeros = tuple(0 for x in range(self.dim))
                zerovector = fd.Constant(zeros)
            else:
                raise NotImplementedError
            for bid in fixed_bids:
                fd.DirichletBC(self.V, zerovector, bid).apply(A)

        #find the nullspace of the Riesz equation, may be None
        #nsp = None
        #if len(fixed_bids) == 0: #may not be a guarantee for weird inner products, suggest to drop this line
        nsp_functions = self.get_nullspace()
        if nsp_functions is not None:
            nsp = fd.VectorSpaceBasis(nsp_functions)
            nsp.orthonormalize()
        else:
            nsp = None

        #the following lines indicate that we want to use firedrake to solve linear systems
        #alternatively, we could use firedrake to define linear systems, but go directly to PETSc for solving
        #the first option requires recasting non-FEControlVector to firedrake control vectors
        params = self.get_params()
        self.ls = fd.LinearSolver(A, solver_parameters=params, nullspace=nsp,
                transpose_nullspace=nsp)
        self.A = fd.as_backend_type(A).mat() #what is this?

    def get_weak_form(self):
        raise NotImplementedError

    def get_nullspace(self):
        return None

    def get_params(self):
        return {
                'ksp_solver': 'gmres', 
                'pc_type': 'lu'
                }

    def riesz_map(self, v, out): # dual to primal
        # expects two FEControlObjects
        if v.fun is None or out.fun is None:
            self.ls.ksp.solve(v.vec, out.vec) # Won't do boundary conditionsd
        self.ls.solve(out.fun, v.fun) #suggestion: force this

    def eval(self, u, v): # inner product in primal space
        # expects two FEControlObjects
        A_u = self.A.createVecLeft()
        self.A.mult(u.vec, A_u)
        return v.vec.dot(A_u)

class HelmholtzInnerProduct(InnerProduct):

    def get_weak_form(self):
        u = fd.TrialFunction(self.V)
        v = fd.TestFunction(self.V)
        return fd.inner(fd.grad(u), fd.grad(v)) * fd.dx + fd.inner(u, v) * fd.dx

class LaplaceInnerProduct(InnerProduct):

    def get_weak_form(self):
        u = fd.TrialFunction(self.V)
        v = fd.TestFunction(self.V)
        return fd.inner(fd.grad(u), fd.grad(v)) * fd.dx

    def get_nullspace(self):
        if self.dim == 2:
            n1 = fd.Function(self.V).interpolate(fd.Constant((1.0, 0.0)))
            n2 = fd.Function(self.V).interpolate(fd.Constant((0.0, 1.0)))
            res = [n1, n2]
        elif self.dim==3:
            n1 = fd.Function(self.V).interpolate(fd.Constant((1.0, 0.0, 0.0)))
            n2 = fd.Function(self.V).interpolate(fd.Constant((0.0, 1.0, 0.0)))
            n3 = fd.Function(self.V).interpolate(fd.Constant((0.0, 0.0, 1.0)))
            res = [n1, n2, n3]
        else:
            raise NotImplementedError
        return res


class InterpolatedInnerProduct(InnerProduct):
    """
    this cannot be correct if the support of the nonFEM basis vector fields is
    larger than the physical domain, or if the computational domains has holes
    that intersect the support of nonFEM basis vector field
    """
    def __init__(self, inner_product, interpolate, restrict):
        self.interpolate = interpolate
        self.restrict = restrict
        self.inner_product = inner_product
        self.A = IT * self.inner_product.A * I
        # set diagonals to one if entire row/column is zero
        self.Aksp = ...

    def riesz_map(self, v, out):
        # v_fd = fd.Function(self.inner_product.V)
        # out_fd = fd.Function(self.inner_product.V)
        # with v_fd.dat.vec as x:
        #     self.interpolate(v, x)
        # self.ls.solve(out_fd, v_fd)
        # #self.inner_product.riesz_map(v_fd, out_fd)
        # out = self.restrict(out_fd)
        self.Aksp.solve(v.vec, out.vec)

        
