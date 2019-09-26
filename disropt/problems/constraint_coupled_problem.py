from .problem import Problem
from ..functions import AbstractFunction


class ConstraintCoupledProblem(Problem):
    """A local part of a constraint-coupled problem. 

    Args:
        objective_function (AbstractFunction, optional): Local objective function. Defaults to None.
        constraints (list, optional): Local constraints. Defaults to None.
        coupling_function (AbstractFunction, optional): Local function contributing to coupling constraints. Defaults to None.

    Attributes:
        objective_function (Function): Objective function to be minimized
        constraints (AbstractSet or Constraint): Local constraints
        coupling_function (Function): Local function contributing to coupling constraints
    """

    coupling_function = None

    def __new__(cls, objective_function: AbstractFunction = None, constraints: list = None,
                coupling_function: AbstractFunction = None):
        instance = object.__new__(cls)
        return instance

    def __init__(self, objective_function: AbstractFunction = None, constraints: list = None,
                 coupling_function: AbstractFunction = None):
        super().__init__(objective_function=objective_function, constraints=constraints)

        if coupling_function is not None:
            self.set_coupling_function(coupling_function)

    def set_coupling_function(self, fn: AbstractFunction):
        """Set the coupling constraint function

        Args:
            fn: coupling constraint function

        Raises:
            TypeError: input must be a AbstractFunction 
        """
        if not isinstance(fn, AbstractFunction):
            raise TypeError("coupling function must be a AbstractFunction")
        self.coupling_function = fn
