import numpy as np
import warnings
from .abstract_function import AbstractFunction
from .affine_form import AffineForm
from .utilities import check_input


class Variable(AffineForm):
    """Variable, basic function

    .. math::

        f(x) = x 

    with :math:`x\\in \\mathbb{R}^{n}`

    Args:
        n (int): dimension of the decision variable: (n,1)

    Raises:
        TypeError: input dimension must be an int
    """

    def __init__(self, n: int):
        if not isinstance(n, int):
            raise TypeError("Input must be an int")
        self.input_shape = (n, 1)
        self.output_shape = (n, 1)

        # affine parameters
        self.A = np.eye(self.input_shape[0])
        self.b = np.zeros(self.output_shape)
        self.fn = self

        self.differentiable = True
        self.affine = True
        super(AffineForm, self).__init__()

        import cvxpy as cvx
        self.cvx_var = cvx.Variable(self.input_shape[0], var_id=0)

    def _expression(self):
        expression = self.__class__.__name__+'{}'.format(self.input_shape)
        return expression
    
    def _to_cvxpy(self):
        return self.cvx_var

    @check_input
    def eval(self, x: np.ndarray) -> np.ndarray:
        return x.reshape(self.output_shape)

    def _extend_variable(self, n_var, axis, pos):
        if axis != 0:
            raise NotImplementedError("Only axis=0 is supported")
        n = self.input_shape[0]
        A = np.zeros((n + n_var, n))
        A[pos:pos+n, :] = np.eye(n)
        return A @ Variable(n + n_var)