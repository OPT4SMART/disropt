import numpy as np
import autograd.numpy as anp
import random
import warnings
from .abstract_function import AbstractFunction
from .constant import Constant
from .affine_form import AffineForm
from .variable import Variable
from .utilities import check_input


class Max(AbstractFunction):
    """Max function (elementwise)

    .. math::

        f(x,y) = \\max(x,y)

    with :math:`x,y: \\mathbb{R}^{n}`.

    Args:
        f1 (AbstractFunction): input function
        f2 (AbstractFunction): input function

    Raises:
        TypeError: input must be a AbstractFunction object
        ValueError: sunctions must have the same input/output shapes
    """

    def __init__(self, f1: AbstractFunction, f2: AbstractFunction):
        if not (isinstance(f1, AbstractFunction) or isinstance(f2, AbstractFunction)):
            raise TypeError("At least one input must be a AbstractFunction object, otherwise use the builtin max operator.")
        
        self.f1 = f1
        self.f2 = f2

        if not isinstance(f1, AbstractFunction):
            if isinstance(f1, (np.ndarray, float, int)):
                v = Variable(f2.input_shape[0])
                A = np.zeros([f2.input_shape[0], f2.output_shape[0]])
                if isinstance(f1, np.ndarray):
                    if f1.shape[1] != 1:
                        f1 = f1.transpose()
                self.f1 = A @ v + f1
            else: 
                raise TypeError("Only AbstractFunction, numpy.ndarray, float and int are supported")
        
        if not isinstance(f2, AbstractFunction):
            if isinstance(f2, (np.ndarray, float, int)):
                v = Variable(f1.input_shape[0])
                A = np.zeros([f1.input_shape[0], f1.output_shape[0]])
                if isinstance(f2, np.ndarray):
                    if f2.shape[1] != 1:
                        f2 = f2.transpose()
                self.f2 = A @ v + f2
            else:
                raise TypeError("Only AbstractFunction, numpy.ndarray, float and int are supported")

        if self.f1.input_shape != self.f2.input_shape or self.f1.output_shape != self.f2.output_shape:
            raise ValueError("Different input/output shapes")
        
        if not (self.f1.is_differentiable and self.f2.is_differentiable):
            warnings.warn(
                'Composition with a nondifferentiable function will lead to an\
                    error when asking for a subgradient')

        self.input_shape = self.f1.input_shape
        self.output_shape = self.f1.output_shape

        self.affine = False
        self.quadratic = False

        super().__init__()

    def _expression(self):
        expression = 'Max({}, {})'.format(self.f1._expression(), self.f2._expression())
        return expression 
    
    def _to_cvxpy(self):
        import cvxpy as cvx
        return cvx.maximum(self.f1._to_cvxpy(), self.f2._to_cvxpy())
    
    def _extend_variable(self, n_var, axis, pos):
        return Max(self.f1._extend_variable(n_var, axis, pos), self.f2._extend_variable(n_var, axis, pos))

    @check_input
    def eval(self, x: np.ndarray) -> np.ndarray:
        return ((self.f1.eval(x) + self.f2.eval(x) + anp.abs(self.f1.eval(x) - self.f2.eval(x)))/2).reshape(self.output_shape)

    # @check_input
    # def jacobian(self, x: np.ndarray, **kwargs) -> np.ndarray:
    #     if not (self.f1.is_differentiable and self.f2.is_differentiable):
    #         warnings.warn("Composition of nondifferentiable functions. The Jacobian may be not correct.")
    #     # else:
    #     j1 = self.f1.jacobian(x, **kwargs)
    #     j2 = self.f2.jacobian(x, **kwargs)
    #     diff = self.f1.eval(x) - self.f2.eval(x)
    #     for elem in range(diff.size):
    #         if diff[elem] == 0:
    #             diff[elem] = random.uniform(-1, 1)
    #     res = j1 + j2 + np.diag(np.sign(diff).flatten()) @ (j1 - j2)
    #     return res/2
