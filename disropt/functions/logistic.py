import numpy as np
import autograd.numpy as anp
import warnings
from .abstract_function import AbstractFunction
from .utilities import check_input


class Logistic(AbstractFunction):
    """Logistic function (elementwise)

    .. math::

        f(x)=\\log(1+e^x)

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
        self.quadratic = False

        super().__init__()

    def _expression(self):
        expression = 'Logistic({})'.format(self.fn._expression())
        return expression

    def _to_cvxpy(self):
        import cvxpy as cvx
        return cvx.logistic(self.fn._to_cvxpy())
    
    def _extend_variable(self, n_var, axis, pos):
        return Logistic(self.fn._extend_variable(n_var, axis, pos))

    @check_input
    def eval(self, x: np.ndarray) -> np.ndarray:
        aux = anp.logaddexp(0, self.fn.eval(x))
        return aux.reshape(self.output_shape)
    
    @check_input
    def _alternative_jacobian(self, x: np.ndarray, **kwargs) -> np.ndarray:
        warnings.simplefilter("error", category=RuntimeWarning)
        if not self.fn.is_differentiable:
            warnings.warn("Composition of non affine functions. The Jacobian/subgradient may be not correct.")

        val = self.fn.eval(x).flatten()

        try:
            p1 = np.diag((1/(1+np.exp(val))).flatten())
            p2 = np.diag((np.exp(val)).flatten())
        except RuntimeWarning:
            # check for overflows
            p1 = np.diag(val)
            p2 = np.diag(val)
            for idx, value in enumerate(val):
                if value > 600:
                    p1[idx, idx] = 0
                    p2[idx, idx] = np.nan_to_num(np.inf)
                else:
                    p1[idx, idx] = 1/(1+np.exp(value))
                    p2[idx, idx] = np.exp(value)
        p3 = self.fn.jacobian(x, **kwargs)
        return p1 @ (p2 @ p3)
