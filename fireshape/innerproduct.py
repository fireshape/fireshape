import firedrake as fd

class InnerProduct(object):

    def __init__(self, mesh, fixed_bids=[]):
        element = mesh.coordinates.function_space().ufl_element()
        self.V = fd.VectorFunctionSpace(mesh, element)

        self.dim = mesh.topological_dimension()
        if self.dim == 2:
            zerovector = fd.Constant((0, 0))
        elif self.dim == 3:
            zerovector = fd.Constant((0, 0, 0))
        else:
            raise NotImplementedError

        a = self.get_weak_form()

        nsp_functions = self.get_nullspace()
        if len(fixed_bids) == 0 and nsp_functions is not None:
            nsp = fd.VectorSpaceBasis(nsp_functions)
            nsp.orthonormalize()
        else:
            nsp = None

        params = self.get_params()

        A = fd.assemble(a, mat_type='aij')
        for bid in fixed_bids:
            fd.DirichletBC(self.V, zerovector, bid).apply(A)

        self.ls = fd.LinearSolver(A, solver_parameters=params, nullspace=nsp,
                transpose_nullspace=nsp)
        A.force_evaluation()
        self.A = fd.as_backend_type(A).mat()

    def get_weak_form(self):
        raise NotImplementedError

    def get_nullspace():
        return None

    def get_params():
        return {
                'ksp_solver': 'cg', 
                'pc_type': 'hypre'
                }

    def riesz_map(self, v, out): # dual to primal
        # expects two firedrake vector objects
        self.ls.solve(out, v)

    def eval(self, u, v): # inner product in primal space
        # expects two firedrake vector objects
        uvec = fd.as_backend_type(u).vec()
        vvec = fd.as_backend_type(v).vec()
        A_u = self.A.createVecLeft()
        self.A.mult(uvec, A_u)
        return vvec.dot(A_u)

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
        # temp = interp.T*v
        # temp2 = interp*temp
        # self.inner_product.riesz_map(temp2, ..)
        # return interpT*...
