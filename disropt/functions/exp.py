import numpy as np
import autograd.numpy as anp
import warnings
from .abstract_function import AbstractFunction
from .utilities import check_input


class Exp(AbstractFunction):
    """Exponential function (elementwise)

    .. math::

        f(x)=e^x

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
        expression = 'Exp({})'.format(self.fn._expression())
        return expression 
    
    def _to_cvxpy(self):
        import cvxpy as cvx
        return cvx.exp(self.fn._to_cvxpy())
    
    def _extend_variable(self, n_var, axis, pos):
        return Exp(self.fn._extend_variable(n_var, axis, pos))

    @check_input
    def eval(self, x: np.ndarray) -> np.ndarray:
        return anp.exp(self.fn.eval(x)).reshape(self.output_shape)

    @check_input
    def _alternative_jacobian(self, x: np.ndarray, **kwargs) -> np.ndarray:
        warnings.simplefilter("error")
        if not self.fn.is_differentiable:
            warnings.warn("Composition of non affine functions. The Jacobian may be not correct.")
        # e^{g(x)} \jac g(x) 
        val = self.fn.eval(x).flatten()
        try:
            p1 = np.diag(np.exp(val))
        except RuntimeWarning:
            # check for overflows
            p2 = np.diag(val)
            for idx, value in enumerate(val):
                if value > 600:
                    p1[idx, idx] = np.nan_to_num(np.inf)
                else:
                    p1[idx, idx] = np.exp(value)
        p2 = self.fn.jacobian(x, **kwargs)
        return p1 @ p2
