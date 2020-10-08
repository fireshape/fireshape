import firedrake as fd
from fireshape import ShapeObjective
from PDEconstraint_beam import CNBeamSolver
import numpy as np


class FakeObjective(ShapeObjective):
    """L2 tracking functional for Poisson problem."""

    def __init__(self, pde_solver: CNBeamSolver, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.pde_solver = pde_solver

    def value_form(self):
        # This is no longer used
        pass
