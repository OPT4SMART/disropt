import numpy as np
import autograd.numpy as anp
import warnings
from .abstract_function import AbstractFunction
from .utilities import check_input


class Log(AbstractFunction):
    """Natural log function (elementwise)

    .. math::

        f(x)=\\log(x)

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
        expression = 'Log({})'.format(self.fn._expression())
        return expression 

    def _to_cvxpy(self):
        import cvxpy as cvx
        return cvx.log(self.fn._to_cvxpy())
    
    def _extend_variable(self, n_var, axis, pos):
        return Log(self.fn._extend_variable(n_var, axis, pos))

    @check_input
    def eval(self, x: np.ndarray) -> np.ndarray:
        return anp.log(self.fn.eval(x)).reshape(self.output_shape)

    @check_input
    def _alternative_jacobian(self, x: np.ndarray, **kwargs) -> np.ndarray:
        warnings.simplefilter("error")
        if not self.fn.is_differentiable:
            warnings.warn("Composition of non affine functions. The Jacobian may be not correct.")

        # (1)/(fn(x)) @ \Jac fn(x) 
        val = self.fn.eval(x).flatten()
        
        try:
            p1 = np.diag(1/val)
        except RuntimeWarning:
            # check for overflows
            p1 = np.diag(val)
            for idx, value in enumerate(val):
                if value > 600:
                    p1[idx, idx] = 0
                else:
                    p1[idx, idx] = 1/(value)

        p2 = self.fn.jacobian(x, **kwargs)
        return p1 @ p2
