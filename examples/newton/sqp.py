from firedrake.petsc import PETSc
import numpy as np
import firedrake as fd

class SQP(object):

    def __init__(self, q, J, e, c=None):
        u = e.solution
        v = e.solution_adj
        F = fd.replace(e.F, {e.F.arguments()[0]: v})
        L = J.value_form() + F
        Q = q.controlspace

        self.q = q
        self.u = u
        self.v = v
        self.F = F
        self.L = L
        self.Q = Q
        self.J = J
        self.c = c
        self.e = e

        with q.fun.dat.vec_wo as _q:
            q_lsize = _q.getLocalSize()
            q_gsize = _q.getSize()
        with u.dat.vec_wo as _u:
            u_lsize = _u.getLocalSize()
            u_gsize = _u.getSize()
        with v.dat.vec_wo as _v:
            v_lsize = _v.getLocalSize()
            v_gsize = _v.getSize()
        
        self.isq = PETSc.IS().createGeneral(list(range(0, q_lsize)))
        self.isu = PETSc.IS().createGeneral(list(range(q_lsize, q_lsize+u_lsize)))
        self.isv = PETSc.IS().createGeneral(list(range(q_lsize+u_lsize, q_lsize+u_lsize+v_lsize)))

        lsize = q_lsize+u_lsize+v_lsize
        gsize = q_gsize+u_gsize+v_gsize
        functions_to_vec = self.functions_to_vec
        vec_to_functions = self.vec_to_functions
        sqp_class = self
        X = fd.SpatialCoordinate(self.u.ufl_domain())

        def stokes_riesz_map(up, vq):
            V = up.function_space()
            u, p = fd.TrialFunctions(V)
            v, q = fd.TestFunctions(V)
            A = fd.assemble(fd.inner(fd.grad(u), fd.grad(v)) * fd.dx + fd.inner(p,q) * fd.dx)
            fd.solve(A, vq, up, bcs=fd.homogenize(e.bcs))

        class MatrixAction(object):

            def mult(self, mat, x, y):
                q_x = q.clone()
                u_x = u.copy(deepcopy=True)
                v_x = v.copy(deepcopy=True)
                Tq_x = fd.Function(Q.V_m)

                q_y = q.clone()
                u_y = u.copy(deepcopy=True)
                v_y = v.copy(deepcopy=True)
                Tq_y = fd.Function(Q.V_m)

                sqp_class.vec_to_functions(x, q_x, u_x, v_x)

                Q.interpolate(q_x, Tq_x)
                Tq_y += fd.assemble(fd.derivative(fd.derivative(L, X, Tq_x), X))
                Tq_y += fd.assemble(fd.derivative(fd.derivative(L, u, u_x), X))
                Tq_y += fd.assemble(fd.derivative(fd.derivative(L, v, v_x), X))
                Q.restrict(Tq_y, q_y)
                if c is not None:
                    temp = q_y.clone()
                    c.hessVec2(temp, q_x, None, None)
                    q_y.plus(temp)

                u_y *= 0
                u_y += fd.assemble(fd.derivative(fd.derivative(L, X, Tq_x), u))
                u_y += fd.assemble(fd.derivative(fd.derivative(L, u, u_x), u))
                u_y += fd.assemble(fd.derivative(fd.derivative(L, v, v_x), u))

                v_y *= 0
                v_y += fd.assemble(fd.derivative(fd.derivative(L, X, Tq_x), v))
                v_y += fd.assemble(fd.derivative(fd.derivative(L, u, u_x), v))
                v_y += fd.assemble(fd.derivative(fd.derivative(L, v, v_x), v))
                sqp_class.apply_bcs(q_y, u_y, v_y)

                sqp_class.functions_to_vec(q_y, u_y, v_y, y)

        class PreconditionerAction(object):

            def apply(self, pc, x, y):
                q_x = q.clone()
                u_x = u.copy(deepcopy=True)
                v_x = v.copy(deepcopy=True)

                sqp_class.vec_to_functions(x, q_x, u_x, v_x)
                q_x.apply_riesz_map()
                u_y = u_x.copy(deepcopy=True)
                v_y = v_x.copy(deepcopy=True)
                stokes_riesz_map(u_x, u_y)
                stokes_riesz_map(v_x, v_y)

                sqp_class.apply_bcs(q_x, u_y, v_y)
                sqp_class.functions_to_vec(q_x, u_y, v_y, y)


        mat = PETSc.Mat()
        mat.createPython(((lsize, gsize), (lsize, gsize)), MatrixAction())
        mat.setUp()
        self.mat = mat

        ksp_sqp = PETSc.KSP()
        ksp_sqp.create()
        ksp_sqp.setOperators(mat)
        pc = ksp_sqp.pc
        pc.setType(pc.Type.PYTHON)
        pc.setPythonContext(PreconditionerAction())

        ksp_sqp.setOptionsPrefix("sqp_")
        opts = PETSc.Options()
        opts["sqp_ksp_type"] = "minres"
        # opts["sqp_ksp_monitor"] = None
        opts["sqp_ksp_monitor_true_residual"] = None
        opts["sqp_ksp_atol"] = 1e-6
        opts["sqp_ksp_rtol"] = 1e-2
        ksp_sqp.setFromOptions()
        ksp_sqp.setUp()
        self.ksp_sqp = ksp_sqp

    def functions_to_vec(self, q, u, v, quv):
        q.vec_ro().copy(quv.getSubVector(self.isq))
        with u.dat.vec_ro as _u:
            _u.copy(quv.getSubVector(self.isu))
        with v.dat.vec_ro as _v:
            _v.copy(quv.getSubVector(self.isv))

    def vec_to_functions(self, quv, q, u, v):
        quv.getSubVector(self.isq).copy(q.vec_wo())
        with u.dat.vec_wo as _u:
            quv.getSubVector(self.isu).copy(_u)
        with v.dat.vec_wo as _v:
            quv.getSubVector(self.isv).copy(_v)

    def apply_bcs(self, q, u, v):
        for bc in fd.homogenize(self.e.bcs):
            bc.apply(u)
            bc.apply(v)
        qbc = fd.DirichletBC(q.fun.function_space(), 0, [1, 2, 3])
        qbc.apply(q.fun)

    def get_dL(self):

        dJq = self.q.clone()
        temp = self.q.clone()

        self.J.derivative(dJq)
        X = fd.SpatialCoordinate(self.u.ufl_domain())
        fd.assemble(fd.derivative(self.F, X, fd.TestFunction(self.J.V_m)), tensor=self.J.deriv_m,
                    form_compiler_parameters=self.J.params)
        temp.from_first_derivative(self.J.deriv_r)
        dLq = dJq
        dLq.plus(temp)
        if self.c is not None:
            self.c.derivative(temp)
            dLq.plus(temp)

        dLu = fd.assemble(fd.derivative(self.L, self.u))
        dLv = fd.assemble(fd.derivative(self.L, self.v))
        dL = self.mat.createVecRight()
        self.apply_bcs(dLq, dLu, dLv)
        print("dLq", dLq.norm())
        print("dLu", fd.norm(dLu))
        print("dLv", fd.norm(dLv))
        self.functions_to_vec(dLq, dLu, dLv, dL)
        print("dL", dL.norm())
        return dL

    def step(self):
        dL = self.get_dL()
        dL *= -1
        sol = self.mat.createVecRight()
        self.ksp_sqp.solve(dL, sol)
        dq = self.q.clone()
        du = self.u.copy(deepcopy=True)
        dv = self.v.copy(deepcopy=True)
        self.vec_to_functions(sol, dq, du, dv)
        self.q.plus(dq)
        self.e.solution += du
        self.e.solution_adj += dv
        self.J.update(self.q, None, 1)


    def taylor_test(self):

        self.Q.update_domain(self.q)
        eps = 0.1
        dq = self.q.clone()
        dq.scale(0)
        dq.plus(self.q)
        dq.scale(eps)
        du = self.u.copy(deepcopy=True)
        du *= eps
        dv = self.v.copy(deepcopy=True)
        dv *= eps

        self.apply_bcs(dq, du, dv)
        fd.File("du.pvd").write(du.split()[0])
        fd.File("dp.pvd").write(du.split()[1])

        print("dq", dq.norm())
        print("du", fd.norm(du))
        print("dv", fd.norm(dv))

        dquv = self.mat.createVecRight()
        self.functions_to_vec(dq, du, dv, dquv)
        res = self.mat.createVecRight()
        self.mat.mult(dquv, res)

        dL0 = self.get_dL()
        print("dL0", dL0.norm())
        # self.q.plus(dq)
        # print("q", self.q.norm())
        self.Q.update_domain(self.q)
        self.u += du
        self.v += dv
        dL1 = self.get_dL()
        print("dL1", dL1.norm())
        err = dL0 + res - dL1


        print("Taylor test", ((err)/eps).norm())

        print("dq", dq.norm())
        print("du", fd.norm(du))
        print("dv", fd.norm(dv))
        self.vec_to_functions(err, dq, du, dv)
        print("err", err.norm())
        print("err q", dq.norm())
        print("err u", fd.norm(du))
        print("err v", fd.norm(dv))
        fd.File("errq.pvd").write(dq.fun)
        fd.File("erru.pvd").write(du.split()[0])
        fd.File("errp.pvd").write(du.split()[1])
        fd.File("errv.pvd").write(dv.split()[0])
        fd.File("errq.pvd").write(dv.split()[1])
