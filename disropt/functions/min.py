import numpy as np
import warnings
from .abstract_function import AbstractFunction
from .utilities import check_input
from .max import Max


class Min(Max):
    """Min function (elementwise)

    .. math::

        f(x,y) = \\min(x,y)

    with :math:`x,y: \\mathbb{R}^{n}`.

    Args:
        f1 (AbstractFunction): input function
        f2 (AbstractFunction): input function

    Raises:
        ValueError: input must be a AbstractFunction object
        ValueError: sunctions must have the same input/output shapes
    """

    def __init__(self, f1: AbstractFunction, f2: AbstractFunction):
        super(Min, self).__init__(-f1, -f2)

    def _to_cvxpy(self):
        import cvxpy as cvx
        return cvx.minimum(self.f1._to_cvxpy(), self.f2._to_cvxpy())
    
    def _extend_variable(self, n_var, axis, pos):
        return Min(-self.f1._extend_variable(n_var, axis, pos), -self.f2._extend_variable(n_var, axis, pos))

    @check_input
    def eval(self, x: np.ndarray) -> np.ndarray:
        return -super(Min, self).eval(x)

    # @check_input
    # def jacobian(self, x: np.ndarray, **kwargs) -> np.ndarray:
    #     return -super(Min, self).jacobian(x, **kwargs)

