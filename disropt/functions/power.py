import numpy as np
import autograd.numpy as anp
import warnings
from .abstract_function import AbstractFunction
from .utilities import check_input


class Power(AbstractFunction):
    """Power function (elementwise)

    .. math::

        f(x)= x^\\alpha

    with :math:`x: \\mathbb{R}^{n}`, :math:`\\alpha: \\mathbb{R}`.

    Args:
        fn (AbstractFunction): input function

    Raises:
        TypeError: input must be a AbstractFunction object
    """

    # TODO Recast as square if exponent is 2
    def __init__(self, fn: AbstractFunction, exponent: float):
        if not isinstance(fn, AbstractFunction):
            raise TypeError("Input must be a AbstractFunction object")

        if not fn.is_differentiable:
            pass
            # warnings.warn(
            #     'Composition with a nondifferentiable function will lead to an\
            #         error when asking for a subgradient')
        else:
            self.differentiable = True

        self.fn = fn
        self.exponent = exponent

        self.input_shape = fn.input_shape
        self.output_shape = fn.output_shape

        self.affine = False
        self.quadratic = False

        super().__init__()
        
    def _expression(self):
        expression = 'Power({}, {})'.format(self.fn._expression(), self.exponent)
        return expression 

    def _to_cvxpy(self):
        import cvxpy as cvx
        return cvx.power(self.fn._to_cvxpy(), p=self.exponent)

    @check_input
    def eval(self, x: np.ndarray) -> np.ndarray:
        return anp.power(self.fn.eval(x), self.exponent).reshape(self.output_shape)
    
    def _extend_variable(self, n_var, axis, pos):
        return Power(self.fn._extend_variable(n_var, axis, pos), self.exponent)