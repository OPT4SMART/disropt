import numpy as np
from typing import Union
from .constraints import Constraint
from copy import deepcopy


class ExtendedConstraint(Constraint):
    """Constraint with extended variable

    Args:
        constr (Constraint or list of Constraint): original constraint(s)
        n_var: number of additional variables. Defaults to 1
        axis: axis along which the additional variables are appended. Defaults to -1 (the last valid one)
        pos: position index of the old variable vector. Defaults to 0

    Raises:
        TypeError: fn must be a Constraint object or a list of Constraint objects
        TypeError: n_var must be a positive int
        TypeError: axis must be int
    """

    def __new__(cls, constr: Union[Constraint, list], n_var: int = 1, axis=-1, pos=0):
        if not isinstance(constr, Constraint) and not isinstance(constr, list):
            raise TypeError("fn must be a Constraint object or a list of Constraint objects")

        if isinstance(constr, list) and not all(isinstance(x, Constraint) for x in constr):
            raise TypeError("fn must be a Constraint object or a list of Constraint objects")

        if not isinstance(n_var, int) or n_var <= 0:
            raise TypeError("n_var must be a positive int")

        if not isinstance(axis, int):
            raise TypeError("axis must be int")

        if isinstance(constr, Constraint):
            from ..functions.extended_function import ExtendedFunction
            return Constraint(ExtendedFunction(constr.fn, n_var, axis, pos), constr.sign)
        else:
            return [ExtendedConstraint(x, n_var, axis, pos) for x in constr]
