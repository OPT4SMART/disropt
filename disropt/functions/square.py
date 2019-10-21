import numpy as np
import autograd.numpy as anp
import warnings
from .abstract_function import AbstractFunction
from .utilities import check_input


class Square(AbstractFunction):
    """Square function (elementwise)

    .. math::

        f(x)= x^2

    with :math:`x: \\mathbb{R}^{n}`.

    Args:
        fn (AbstractFunction): input function

    Raises:
        TypeError: input must be a AbstractFunction object
    """

    def __init__(self, fn: AbstractFunction):
        if not isinstance(fn, AbstractFunction):
            raise TypeError("Input must be a AbstractFunction object")

        if not fn.is_differentiable:
            warnings.warn(
                'Composition with a nondifferentiable function will lead to an\
                    error when asking for a subgradient')
        else:
            self.differentiable = True

        self.fn = fn

        self.input_shape = fn.input_shape
        self.output_shape = fn.output_shape

        self.affine = False
        self.quadratic = False # TODO

        super().__init__()
        
    def _expression(self):
        expression = 'Square({})'.format(self.fn._expression())
        return expression 

    def _to_cvxpy(self):
        import cvxpy as cvx
        return cvx.square(self.fn._to_cvxpy())

    @check_input
    def eval(self, x: np.ndarray) -> np.ndarray:
        return anp.power(self.fn.eval(x), 2).reshape(self.output_shape)
    
    def _extend_variable(self, n_var, axis, pos):
        return Square(self.fn._extend_variable(n_var, axis, pos))

    # @check_input
    # def jacobian(self, x: np.ndarray, **kwargs) -> np.ndarray:
    #     if not self.fn.is_differentiable:
    #         warnings.warn("Composition of non affine functions. The Jacobian may be not correct.")
    #     # 2 fn(x) \jac fn(x)
    #     val = self.fn.eval(x)
    #     p1 = 2 * np.diag(val.flatten())
    #     p2 = self.fn.jacobian(x, **kwargs)
    #     return p1 @ p2
