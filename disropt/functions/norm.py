import numpy as np
import autograd.numpy as anp
import random
import warnings
from typing import Union
from .abstract_function import AbstractFunction
from .utilities import check_input


class Norm(AbstractFunction):
    """Norm of a function (supporte norms are 1, 2, inf)

    .. math::

        f(x)=\\|x\\|

    with :math:`x: \\mathbb{R}^{n}`.

    Args:
        fn (AbstractFunction): input function
        order (int, optional): order of the norm. Can be 1, 2 or np.inf. Defaults to 2.

    Raises:
        TypeError: input must be a function object
        NotImplementedError: only 1, 2 and inf norms are currently supported
    """

    def __init__(self, fn: AbstractFunction, order: Union[int, float] = None, axis: int = None):
        if not isinstance(fn, AbstractFunction):
            raise TypeError("Input must be a AbstractFunction object")
        self.fn = fn
        if not fn.is_differentiable:
            warnings.warn(
                'Composition with a nondifferentiable function will lead to an\
                     error when asking for a subgradient')

        if not fn.is_affine:
            warnings.warn(
                'Composition with a non affine function will lead to an error \
                    when asking for a subgradient')

        if order not in [1, 2, np.inf, None]:
            raise NotImplementedError
        self.order = order
        self.axis = axis
        self.input_shape = fn.input_shape
        self.last_input_shape = fn.output_shape
        self.output_shape = (1, 1)

        self.differentiable = False
        self.affine = False
        self.quadratic = False

        super().__init__()

    def _expression(self):
        expression = 'Norm({}, order={})'.format(self.fn._expression(), self.order)
        return expression
    
    def _to_cvxpy(self):
        import cvxpy as cvx
        if self.order == 2 or self.order == None:
            return cvx.norm(self.fn._to_cvxpy())
        
        if self.order == 1:
            return cvx.norm1(self.fn._to_cvxpy())

        if self.order == np.inf:
            return cvx.norm_inf(self.fn._to_cvxpy())
    
    def _extend_variable(self, n_var, axis, pos):
        return Norm(self.fn._extend_variable(n_var, axis, pos), self.order, self.axis)

    @check_input
    def eval(self, x: np.ndarray) -> np.ndarray:
        return anp.linalg.norm(self.fn.eval(x), ord=self.order, axis=self.axis).reshape(self.output_shape)

    def _alternative_jacobian(self, x: np.ndarray, **kwargs) -> np.ndarray:
        if self.order in [1, 2, np.inf, None]:
            if not self.fn.is_differentiable:
                warnings.warn("Composition of non affine functions. The Jacobian may be not correct.")
            if self.last_input_shape[1] != 1:
                raise NotImplementedError
            p_subg = np.zeros(self.last_input_shape)
            pt = self.fn.eval(x)
            if self.order == 1:
                for i in range(self.last_input_shape[0]):
                    if pt[i] != 0:
                        p_subg[i] = np.sign(pt[i])
                    else:
                        p_subg[i] = random.uniform(-1, 1)
            if self.order == 2 or self.order == None:
                for i in range(self.last_input_shape[0]):
                    if pt[i] != 0:
                        p_subg[i] = pt[i] / np.linalg.norm(pt, ord=2)
                    else:
                        p_subg[i] = random.uniform(-1, 1)
            if self.order == np.inf:
                n1 = np.linalg.norm(pt, ord=1, axis=1).reshape(self.input_shape)
                idx = np.argmax(n1, axis=1)
                w1 = np.random.rand(self.input_shape)
                w2 = np.zeros(self.input_shape)
                w2 += w1
                w2[idx] = 0
                w = w1 - w2
                w = w / sum(w)
                for i in idx:
                    if pt[i] != 0:
                        p_subg[i] = w[i] * np.sign(pt[i])
                    else:
                        p_subg[i] = w[i] * random.uniform(-1, 1)
            subg = p_subg.transpose() @ self.fn.jacobian(x, **kwargs)
            return subg
        else:
            raise NotImplementedError
