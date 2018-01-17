import firedrake as fd

class InnerProduct(object):

    def __init__(self, mesh, fixed_bids=[]):
        element = mesh.coordinates.function_space().ufl_element()
        self.V = fd.FunctionSpace(mesh, element)

        self.dim = mesh.topological_dimension()
        if self.dim == 2:
            zerovector = fd.Constant((0, 0))
        elif self.dim == 3:
            zerovector = fd.Constant((0, 0, 0))
        else:
            raise NotImplementedError

        a = self.get_weak_form()


        nsp = None
        if len(fixed_bids) == 0:
            nsp_functions = self.get_nullspace()
            if nsp_functions is not None:
                nsp = fd.VectorSpaceBasis(nsp_functions)
                nsp.orthonormalize()

        params = self.get_params()

        A = fd.assemble(a, mat_type='aij')
        for bid in fixed_bids:
            fd.DirichletBC(self.V, zerovector, bid).apply(A)

        self.ls = fd.LinearSolver(A, solver_parameters=params, nullspace=nsp,
                transpose_nullspace=nsp)
        self.A = fd.as_backend_type(A).mat()

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
        # expects two firedrake vector objects
        if v.fun is None or out.fun is None:
            self.ls.ksp.solve(v.vec, out.vec) # Won't do boundary conditions
        self.ls.solve(out.fun, v.fun)

    def eval(self, u, v): # inner product in primal space
        # expects two firedrake vector objects
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


class InterpolatingInnerProduct(InnerProduct):

    def __init__(self, inner_product, interp):
        self.interp = interp
        self.inner_product = inner_product


    def riesz_map(self, v, out):
        # temp = interp*v
        # self.inner_product.riesz_map(temp2, ..)
        # return interpT*...
        pass
