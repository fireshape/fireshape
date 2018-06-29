import fireshape as fs
from firedrake import split, FacetNormal, inner, ds, Constant, dx, \
    div, tr, grad, derivative, SpatialCoordinate, sym, sqrt, SubDomainData, \
    conditional, ge


class DissipatedEnergy(fs.ShapeObjective):

    def __init__(self, pde_solver, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.pde_solver = pde_solver

    def value_form(self):
        (u, p) = split(self.pde_solver.solution)
        nut = self.pde_solver.nut()
        nu = self.pde_solver.nu
        form = (nut + nu) * inner(grad(u), sym(grad(u))) * dx
        # form = inner(grad((nut + nu) * u), sym(grad(u))) * dx
        return form

    def derivative_form(self, deformation):
        return derivative(self.value_form(), SpatialCoordinate(deformation.ufl_domain()), deformation)


class EnergyRecovery(fs.ShapeObjective):

    def __init__(self, pde_solver, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.pde_solver = pde_solver

    def value_form(self):
        (u, p) = split(self.pde_solver.solution)
        n = FacetNormal(self.Q.mesh_m)
        return -inner(u, n) * (p + 0.5 * inner(u, u)**2) * ds

    def derivative_form(self, deformation):
        return derivative(self.value_form(), SpatialCoordinate(deformation.ufl_domain()), deformation)


class PressureRecovery(fs.ShapeObjective):

    def __init__(self, pde_solver, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.pde_solver = pde_solver

    def value_form(self):
        (u, p) = split(self.pde_solver.solution)
        return -div(u*p) * dx

    def derivative_form(self, deformation):
        return derivative(self.value_form(), SpatialCoordinate(deformation.ufl_domain()), deformation)
