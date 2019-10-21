import numpy as np
import autograd.numpy as anp
import warnings
from typing import Union
from .abstract_function import AbstractFunction
from .utilities import check_input
from .norm import Norm


class SquaredNorm(AbstractFunction):
    """Squared norm (supporte norms are 1, 2, inf)
     
    .. math::

        f(x)=\\|x\\|^2

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

        # if not fn.is_differentiable:
        #     warnings.warn(
        #         'Composition with a nondifferentiable function will lead to an\
        #             error when asking for a subgradient')

        # if not fn.is_affine:
        #     warnings.warn(
        #         'Composition with a non affine function will lead to an error \
        #             when asking for a subgradient')

        self.fn = fn
        if order not in [1, 2, np.inf, None]:
            raise NotImplementedError
        if order == 2:
            self.differentiable = True

        self.order = order
        self.axis = axis
        self.input_shape = fn.input_shape
        self.output_shape = (1, 1)

        self.affine = False
        self.quadratic = False # TODO

        super().__init__()

    def _expression(self):
        expression = 'SquaredNorm({})'.format(self.fn._expression())
        return expression 

    def _to_cvxpy(self):
        import cvxpy as cvx
        return cvx.square(cvx.norm(self.fn._to_cvxpy()))

    @check_input
    def eval(self, x: np.ndarray) -> np.ndarray:
        # TODO: order
        return anp.power(anp.linalg.norm(self.fn.eval(x), ord=self.order, axis=self.axis), 2).reshape(self.output_shape)
    
    def _extend_variable(self, n_var, axis, pos):
        return SquaredNorm(self.fn._extend_variable(n_var, axis, pos), self.order, self.axis)

    @check_input
    def _alternative_jacobian(self, x: np.ndarray, **kwargs) -> np.ndarray:
        if self.order in [1, 2, np.inf, None]:
            if not self.fn.is_differentiable:
                warnings.warn("Composition of nondifferentiable functions. The Jacobian may be not correct.")
                # raise ValueError(
                #     "Composition of nondifferentiable functions. \
                #         No subgradients available.")
            if not self.fn.is_affine:
                warnings.warn("Composition of non affine functions. The Jacobian may be not correct.")
                # raise ValueError(
                #     "Composition of non affine function. \
                #         No subgradients available.")
            if self.order in [2, None]: 
                return 2 * self.fn.eval(x).transpose() @ self.fn.jacobian(x, **kwargs)
            else:
                norm = Norm(self.fn, order=self.order)
                return 2 * norm.jacobian(x, **kwargs) * \
                    np.linalg.norm(self.fn.eval(x), ord=self.order)
        else:
            raise NotImplementedError
    
