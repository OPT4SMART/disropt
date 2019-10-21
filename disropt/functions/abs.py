import numpy as np
import autograd.numpy as anp
import random
import warnings
from typing import Union
from .abstract_function import AbstractFunction
from .utilities import check_input


class Abs(AbstractFunction):
    """Absolute value (element-wise)

    .. math::

        f(x)=|x|

    with :math:`x: \\mathbb{R}^{n}`.

    Args:
        fn (AbstractFunction): input function

    Raises:
        TypeError: input must be a function object
        NotImplementedError: only 1, 2 and inf norms are currently supported
    """

    def __init__(self, fn: AbstractFunction):
        if not isinstance(fn, AbstractFunction):
            raise TypeError("Input must be a AbstractFunction object")
        self.fn = fn
        # if not fn.is_differentiable:
        #     warnings.warn(
        #         'Composition with a nondifferentiable function will lead to an\
        #              error when asking for a subgradient')

        # if not fn.is_affine:
        #     warnings.warn(
        #         'Composition with a non affine function will lead to an error \
        #             when asking for a subgradient')

        self.input_shape = fn.input_shape
        self.output_shape = fn.output_shape

        self.differentiable = False
        self.affine = False
        self.quadratic = False

        super().__init__()

    def _expression(self):
        expression = 'Abs({})'.format(self.fn._expression())
        return expression

    def _to_cvxpy(self):
        import cvxpy as cvx
        return cvx.abs(self.fn._to_cvxpy())

    def _extend_variable(self, n_var, axis, pos):
        return Abs(self.fn._extend_variable(n_var, axis, pos))

    @check_input
    def eval(self, x: np.ndarray) -> np.ndarray:
        return anp.abs(self.fn.eval(x)).reshape(self.output_shape)
