class PdeConstraint(object):
    """
    Base class for PdeConstraint.
    """

    def __init__(self):
        """Set counters of state/adjoint solves to 0."""
        self.num_solves = 0

    def solve(self):
        """Abstract method that solves state equation."""
        self.num_solves += 1
