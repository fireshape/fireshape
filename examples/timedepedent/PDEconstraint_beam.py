from firedrake import *
from fireshape import PdeConstraint


class CNBeamSolver(PdeConstraint):
    def __init__(self, mesh_m):
        super().__init__()
        self.mesh_m = mesh_m

        V = VectorFunctionSpace(self.mesh_m, "CG",1)
        W = V*V
        self.W = W
        Tmax = 1.0; #10.0
        dt = Constant(2e-2);
        self.dt = dt
        t = Constant(0)
        self.t = t
        self.Ns = int(Tmax/float(dt))
        rho = Constant(1.0)

        E,nu= 1000.0,0.3
        self.mu, self.lmbda = Constant(E/(2*(1 + nu))), Constant(E*nu/((1 + nu)*(1 - 2*nu)))

        sol = Function(W, name = "u^{n}")
        u0 = Function(W, name = "u^{n-1}")
        tf = TestFunction(W)

        t0,t1 = split(tf)
        u,v = split(sol)
        u_0,v_0 = split(u0)

        self.bcs = [DirichletBC(W.sub(0), Constant((0,0)), 1)] 
        pres = conditional(t<0.5, Constant(1e-2)*t, 0)*as_vector([0,1])

        # CRANK-Nicholson Residual
        F1 = inner(u-u_0,t0)*dx - Constant(0.5)*dt*inner(v+v_0,t0)*dx
        F2 = inner(v-v_0,t1)*dx + dt/rho * inner(self.sigma(Constant(0.5)*(u+u_0)),self.eps(t1))*dx
        F3 = dt/rho*dot(pres,t1)*ds(3)

        self.R = F1+F2+F3
        # SolverParameters
        self.sp ={"mat_type": "aij",
              #"snes_monitor": None,
              "snes_rtol": 1e-9,
              "snes_atol": 1e-8,
              "snes_linesearch_type": "l2",
              "ksp_type": "preonly",
              "pc_type" : "lu",
              "pc_factor_mat_solver_type": "mumps"} 

        self.test_function=tf
        self.sol = sol
        self.u0 = u0

    def eps(self,r):
        return sym(grad(r))

    def sigma(self,r):
        return self.lmbda*tr(self.eps(r))*Identity(2)+2*self.mu*self.eps(r)

    def solve(self):
        super().solve()
        count = self.num_solves
        self.J = 0.0
        t = self.t
        dt = self.dt
        Ns = self.Ns
        sol = self.sol
        u0 = self.u0
        outfile = File("output/test%d.pvd"%count)
        for i in range(Ns):
            print("Iteration %d"%i)
            solve(self.R==0,self.sol,bcs = self.bcs, solver_parameters = self.sp)
            t.assign(float(t)+float(dt))
            u0.assign(sol)
            u,v = sol.split()
            u.rename("Displacement")
            v.rename("Velocity")
            self.J += assemble(Constant(1e3)*inner(self.sigma(u),self.eps(u))*dx)
            outfile.write(u,v)



if __name__ == "__main__":
    mesh = RectangleMesh(40,4,1,0.1)
    e = CNBeamSolver(mesh) 
    e.solve()
