import numpy  as np
import warnings
from .abstract_function import AbstractFunction
from .utilities import check_input

# DEPRECATED

class Constant(AbstractFunction):
    """Constant, basic function

    .. math::

        f(x)=c

    with :math:`c\\in \\mathbb{R}^{n}`.

    Args:
        c (numpy.ndarray): constant value (with shape (n, 1))
    """

    differentiable = True
    affine = True

    def __init__(self, c: np.ndarray):
        if not isinstance(c, np.ndarray):
            raise TypeError("Input must be an numpy.ndarray")
        if len(c.shape) != 2 or c.shape[1] != 1:
            raise ValueError("Input must be an numpy.ndarray with shape (Any, 1)")
        self.value = c
        self.input_shape = c.shape
        self.output_shape = c.shape

    #@check_input
    def eval(self, x: np.ndarray=None) -> np.ndarray:
        return self.value.reshape(self.output_shape)

    #@check_input
    def jacobian(self, x: np.ndarray, **kwargs) -> np.ndarray:
        return np.zeros([self.input_shape[0], x.shape[0]])
