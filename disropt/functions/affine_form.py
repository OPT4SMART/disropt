import numpy as np
import warnings
from typing import Union
from .abstract_function import AbstractFunction
from .quadratic_form import QuadraticForm
from .utilities import check_input


class AffineForm(AbstractFunction):
    """Makes an affine transformation

    .. math::

        f(x)=\\langle A, x\\rangle + b=A^\\top x + b

    with :math:`A\\in \\mathbb{R}^{n\\times m}`, :math:`b\\in \\mathbb{R}^{m}`
    and :math:`x: \\mathbb{R}^{n}`. It can also be instantiated as::

        A @ x + b

    Args:
        fn (AbstractFunction): input function
        A (numpy.ndarray): input matrix
        b (numpy.ndarray): input bias

    Raises:
        TypeError: first argument must be a AbstractFunction object
        TypeError: second argument must be numpy.ndarray
        ValueError: the number of columns of A must be equal to the number of
        rows of the output of fn
    """

    def __init__(self, fn: AbstractFunction, A: np.ndarray = None, b: np.ndarray = None):
        if not isinstance(fn, AbstractFunction):
            raise TypeError("First argument must be a AbstractFunction object")

        if not fn.is_differentiable:
            warnings.warn(
                'Composition with a nondifferentiable function will lead to \
                    an error when asking for a jacobian or a subgradient')
            self.differentiable = False
        else:
            self.differentiable = True

        if A is not None:
            if not isinstance(A, np.ndarray):
                raise TypeError("Second argument must be a numpy.ndarray")
            if A.shape[0] != fn.output_shape[0]:
                raise ValueError(
                    "Dimension mismatch. Input matrix must have shape {}".format(
                        (fn.output_shape[0], "Any")))
        else:
            A = np.eye(fn.output_shape[0])

        if b is not None:
            if not isinstance(b, np.ndarray):
                if isinstance(b, (int, float)):
                    b = np.ndarray(b).reshape(1, 1)
                else:
                    raise TypeError("Second argument must be a numpy.ndarray")
            if b.shape[0] != A.shape[1]:
                raise ValueError(
                    "Dimension mismatch. Input bias must have shape {}".format(
                        (A.shape[1], 1)))
        else:
            b = np.zeros([A.shape[1], 1])

        if fn.is_affine:
            from .variable import Variable
            self.affine = True
            fn_A, fn_b = fn.get_parameters()
            # A @ (B @ x) = A^T B^T x = (BA)^T x = (B @ A) @ x
            self.A = fn_A @ A
            self.b = b + A.transpose() @ fn_b
            self.fn = Variable(fn.input_shape[0])
        else:
            self.affine = False
            self.A = A
            self.b = b
            self.fn = fn

        self.input_shape = self.fn.input_shape
        self.output_shape = (A.shape[1], self.fn.output_shape[1])

        super().__init__()

    def _expression(self):
        expression = 'AffineForm({}, A, b)'.format(self.fn._expression())
        return expression

    def _to_cvxpy(self):
        return self.A.transpose() @ self.fn._to_cvxpy() + self.b.flatten()
    
    def _extend_variable(self, n_var, axis, pos):
        return AffineForm(self.fn._extend_variable(n_var, axis, pos), self.A, self.b)

    def get_parameters(self):
        if self.is_affine:
            return self.A, self.b

    def __neg__(self):
        A = -1 * self.A
        b = -1 * self.b
        return AffineForm(self.fn, A, b)

    def __add__(self, other):
        if isinstance(other, (int, float, np.ndarray)):
            A = self.A
            b = self.b + other
            return AffineForm(self.fn, A, b)
        elif isinstance(other, AbstractFunction):
            if self.is_affine and other.is_affine:
                other_A, other_b = other.get_parameters()
                if self.A.shape == other_A.shape:
                    A = self.A + other_A
                    b = self.b + other_b
                    return AffineForm(self.fn, A, b)
                else:
                    raise ValueError("Incompatible dimensions")
            if self.is_affine and other.is_quadratic:
                other_P, other_q, other_r = other.get_parameters()
                if other_q.shape == self.A.shape:
                    P = other_P
                    q = self.A + other_q
                    r = self.b + other_r
                    return QuadraticForm(self.fn, P, q, r)
                else:
                    raise ValueError("Incompatible dimensions")
            else:
                return super().__add__(other)
        else:
            raise TypeError

    def __radd__(self, other):
        return self.__add__(other)

    def __iadd__(self, other):
        return self.__add__(other)

    def __sub__(self, other):
        if isinstance(other, (int, float, np.ndarray)):
            A = self.A
            b = self.b - other
            return AffineForm(self.fn, A, b)
        elif isinstance(other, AbstractFunction):
            if self.is_affine and other.is_affine:
                other_A, other_b = other.get_parameters()
                if self.A.shape == other_A.shape:
                    A = self.A - other_A
                    b = self.b - other_b
                    return AffineForm(self.fn, A, b)
                else:
                    raise ValueError("Incompatible dimensions")
            if self.is_affine and other.is_quadratic:
                other_P, other_q, other_r = other.get_parameters()
                if other_q.shape == self.A.shape:
                    P = -other_P
                    q = self.A - other_q
                    r = self.b - other_r
                    return QuadraticForm(self.fn, P, q, r)
                else:
                    raise ValueError("Incompatible dimensions")
            else:
                return super().__sub__(other)
        else:
            raise TypeError

    def __rsub__(self, other):
        return self.__neg__().__add__(other)

    def __isub__(self, other):
        return self.__sub__(other)

    def __mul__(self, other):
        if isinstance(other, (int, float)):
            A = other * self.A
            b = other * self.b
            return AffineForm(self.fn, A, b)
        else:
            return super().__mul__(other)

    def __rmul__(self, other: Union[int, float]):
        return self.__mul__(other)

    def __imul__(self, other):
        return self.__mul__(other)

    def __matmul__(self, other):
        # if isinstance(other, (np.ndarray)):
        #     if other.shape == self.output_shape:
        #         # B @ (A @ x) = B^T A^T x = (AB)^T x = (A @ B) @ x
        #         A = self.A @ other
        #         b = other.transpose() @ self.b
        #         return AffineForm(self.fn, A, b)
        #     else:
        #         raise NotImplementedError
        if isinstance(other, AbstractFunction):
            if self.is_affine and other.is_affine:
                # TODO: check
                other_A, other_b = other.get_parameters()
                if self.A.shape == other_A.shape:
                    P = self.A @ other_A.transpose()
                    q = (self.b.transpose() @ other_A.transpose() +
                         other_b.transpose() @ self.A.transpose()).transpose()
                    r = self.b.transpose() @ other_b
                    return QuadraticForm(self.fn, P, q, r)
                else:
                    raise ValueError("Incompatible dimensions")
            else:
                return super().__matmul__(other)
        else:
            return super().__matmul__(other)

    def __rmatmul__(self, other):
        if isinstance(other, np.ndarray):
            # C^T (A^T x + b) = C^T A^T x + C^T b
            if other.shape[0] != self.output_shape[0]:
                raise ValueError("Incompatible shapes")
            else:
                A = self.A @ other
                b = other.transpose() @ self.b
                return AffineForm(self.fn, A, b)
        elif isinstance(other, AbstractFunction):
            from .constant import Constant
            if isinstance(other, Constant):
                # TODO check dimensions
                A = self.A @ other.eval()
                b = other.eval().transpose() @ self.b
                return AffineForm(self.fn, A, b)
            else:
                return super().__rmatmul__(other)
        else:
            raise TypeError

    def __imatmul__(self, other):
        return self.__matmul__(other)

    @check_input
    def eval(self, x: np.ndarray) -> np.ndarray:
        return (self.A.transpose() @ self.fn.eval(x)).reshape(self.output_shape) + self.b

    @check_input
    def _alternative_jacobian(self, x: np.ndarray, **kwargs) -> np.ndarray:
        if not self.fn.is_differentiable:
            warnings.warn("Composition of non affine functions. The Jacobian may be not correct.")

        # A^T \Jac fn(x)
        return self.A.transpose() @ self.fn.jacobian(x, **kwargs)


def aggregate_affine_form(list_of_affine: list) -> AffineForm:
    from .variable import Variable
    input_shape = None
    x = None
    A = None
    b = None
    for item in list_of_affine:
        if not isinstance(item, AffineForm):
            raise TypeError("A list of AffineForm objects is required.")
        if input_shape is None:
            input_shape = item.input_shape
            x = Variable(input_shape[0])

        if item.input_shape != input_shape:
            raise ValueError("All AffineForm objects in the list must have the same input_shape.")

        item_A, item_b = item.get_parameters()
        if A is None:
            A = item_A
            b = item_b
        else:
            A = np.hstack([A, item_A])
            b = np.vstack([b, item_b])
    return AffineForm(x, A, b)
