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
        order (int, optional): order of the norm. Can be 1, 2 or np.inf. Defaults to 2.
    
    Raises:
        TypeError: input must be a function object
        NotImplementedError: only 1, 2 and inf norms are currently supported
    """

    def __init__(self, fn: AbstractFunction, order: Union[int, float]=None, axis: int=None):
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

    @check_input
    def eval(self, x: np.ndarray) -> np.ndarray:
        return anp.abs(self.fn.eval(x)).reshape(self.output_shape)