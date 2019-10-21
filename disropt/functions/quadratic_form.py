import numpy as np
import warnings
from typing import Union
from .abstract_function import AbstractFunction
from .utilities import check_input
from ..utils.utilities import is_pos_def, is_semi_pos_def


class QuadraticForm(AbstractFunction):
    """Quadratic form

    .. math::

        f(x)= x^\\top P x + q^\\top x + r

    with :math:`P\\in \\mathbb{R}^{n\\times n}`, :math:`q\\in \\mathbb{R}^{n}`, :math:`r\\in \\mathbb{R}` and :math:`x: \\mathbb{R}^{n}`.

    Args:
        fn (AbstractFunction): input function
        P (numpy.ndarray, optional): input matrix. Defaults to None (identity).
        q (numpy.ndarray, optional): input vector. Defaults to None (zero).
        r (numpy.ndarray, optional): input bias. Defaults to None (zero).

    Raises:
        TypeError: First argument must be a AbstractFunction object
        TypeError: Second argument must be a numpy.ndarray
        ValueError: Input matrix must be a square matrix
        ValueError: Dimension mismatch. Input matrix must have shape compliant with the function output shape
    """

    def __init__(self, fn: AbstractFunction, P: np.ndarray = None, q: np.ndarray = None, r: np.ndarray = None):
        if not isinstance(fn, AbstractFunction):
            raise TypeError("First argument must be a AbstractFunction object")

        if not fn.is_differentiable:
            warnings.warn(
                'Composition with a nondifferentiable function will \
                    lead to an error when asking for a subgradient')
        else:
            self.differentiable = True

        if P is not None:
            if not isinstance(P, np.ndarray):
                raise TypeError("Second argument must be a numpy.ndarray")
            row, col = P.shape
            if row != col:
                raise ValueError("Input matrix must be a square matrix")
            if P.shape[0] != fn.output_shape[0]:
                raise ValueError(
                    "Dimension mismatch. Input matrix must have shape {}".format(
                        (fn.output_shape[0], fn.output_shape[0])))
            # if not is_semi_pos_def(P):
            #     warnings.warn("Warning, input matrix is not (semi)positive definite")
        else:
            P = np.eye(fn.output_shape[0])

        if q is not None:
            if not isinstance(q, np.ndarray):
                raise TypeError("third argument must be a numpy.ndarray")

            if q.shape != fn.output_shape:
                raise ValueError(
                    "Dimension mismatch. Input vector must have shape {}".format(fn.output_shape))
        else:
            q = np.zeros(fn.output_shape)

        if r is not None:
            if not isinstance(q, (np.ndarray, float, int)):
                raise TypeError("4-th argument must be a int, float or numpy.ndarray")

            if isinstance(r, np.ndarray):
                if r.shape != (1, 1):
                    raise ValueError(
                        "Dimension mismatch. Input bial must have shape (1,1)")
        else:
            r = 0.0

        self.input_shape = fn.input_shape
        self.output_shape = (1, 1)

        if fn.is_affine:
            self.quadratic = True
            # (A^T x + b)^T P (A^T x + b) + q^T(A^T x + b) + r =
            # = x^T A P A^T x + x^T A P b + b^T P A^T x + b^T P b + q^T A^T x + q^T b + r
            # = x^T A P A^T x + (b^T(P + P^T)A^T + q^t A^t) x + b^T P b + q^T b + r
            #       -------     ---------------------------     -------------------
            # = x^T   P     x +             q^t             x +          r
            from .variable import Variable
            A = fn.A
            b = fn.b
            self.P = A @ P @ A.transpose()
            self.q = (b.transpose() @ (P + P.transpose()) @ A.transpose() +
                      q.transpose() @ A.transpose()).transpose()
            self.r = b.transpose() @ P @ b + q.transpose() @ b + r
            self.fn = Variable(self.input_shape[0])

        else:
            self.quadratic = False
            self.fn = fn
            self.P = P
            self.q = q
            self.r = r

        self.affine = False

        super().__init__()

    def _expression(self):
        expression = 'QuadraticForm({}, P, q, r)'.format(self.fn._expression())
        return expression 
    
    def _to_cvxpy(self):
        import cvxpy as cvx
        fn = self.fn._to_cvxpy()
        return cvx.quad_form(fn, self.P) + self.q.transpose() @ fn + self.r.flatten()
    
    def _extend_variable(self, n_var, axis, pos):
        return QuadraticForm(self.fn._extend_variable(n_var, axis, pos), self.P, self.q, self.r)

    def get_parameters(self):
        if self.is_quadratic:
            return self.P, self.q, self.r

    def __neg__(self):  # TODO Check
        P = -1 * self.P
        q = -1 * self.q
        r = -1 * self.r
        return QuadraticForm(self.fn, P, q, r)

    def __add__(self, other):
        if isinstance(other, (int, float, np.ndarray)):
            P = self.P
            q = self.q
            r = self.r + other
            return QuadraticForm(self.fn, P, q, r)
        elif isinstance(other, AbstractFunction):
            if self.is_quadratic and other.is_affine:
                other_A, other_b = other.get_parameters()
                if self.q.shape == other_A.shape:
                    P = self.P
                    q = self.q + other_A
                    r = self.r + other_b
                    return QuadraticForm(self.fn, P, q, r)
                else:
                    raise ValueError("Incompatible dimensions")
            if self.is_quadratic and other.is_quadratic:
                other_P, other_q, other_r = other.get_parameters()
                if self.P.shape == other_P.shape:
                    P = self.P + other_P
                    q = self.q + other_q
                    r = self.r + other_r
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
            P = self.P
            q = self.q
            r = self.r - other
            return QuadraticForm(self.fn, P, q, r)
        elif isinstance(other, AbstractFunction):
            if self.is_quadratic and other.is_affine:
                other_A, other_b = other.get_parameters()
                if self.q.shape == other_A.shape:
                    P = self.P
                    q = self.q - other_A
                    r = self.r - other_b
                    return QuadraticForm(self.fn, P, q, r)
                else:
                    raise ValueError("Incompatible dimensions")
            if self.is_quadratic and other.is_quadratic:
                other_P, other_q, other_r = other.get_parameters()
                if self.P.shape == other_P.shape:
                    P = self.P - other_P
                    q = self.q - other_q
                    r = self.r - other_r
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
            P = other * self.P
            q = other * self.q
            r = other * self.r
            return QuadraticForm(self.fn, P, q, r)
        else:
            return super().__mul__(other)

    def __rmul__(self, other: Union[int, float]):
        return self.__mul__(other)

    def __imul__(self, other):
        return self.__mul__(other)

    @check_input
    def eval(self, x: np.ndarray) -> np.ndarray:
        p1 = self.fn.eval(x).transpose() @ self.P @ self.fn.eval(x)
        p2 = self.q.transpose() @ self.fn.eval(x)
        p3 = self.r
        return (p1 + p2 + p3).reshape(self.output_shape)

    @check_input
    def _alternative_jacobian(self, x: np.ndarray, **kwargs) -> np.ndarray:
        if not self.fn.is_differentiable:
            warnings.warn("Composition of non affine functions. The Jacobian may be not correct.")
        return self.fn.eval(x).transpose().dot((self.P +
                                                self.P.transpose()).dot(
            self.fn.jacobian(x, **kwargs)
        )) + self.q.transpose().dot(self.fn.jacobian(x, **kwargs))

    # @check_input
    # def hessian(self, x: np.ndarray = None, **kwargs) -> np.ndarray:
    #     from .variable import Variable
    #     if isinstance(self.fn, Variable):
    #         return self.P
    #     else:
    #         return NotImplemented
