import numpy as np
from autograd import grad as ag_grad
from autograd import jacobian as ag_jacobian
from autograd import hessian as ag_hessian
from typing import Union
from .utilities import check_input
import warnings


class AbstractFunction:
    """AbstractFunction class. This should be the parent of all specific (objective) functions.

    Attributes:
        input_shape (tuple): shape of the input of the function
        output_shape (tuple): shape of the output of the function
        differentiable (bool): True if the function is differentiable
        affine (bool): True if the function is affine
        quadratic (bool): True if the function is quadratic
    """
    # For forcing __r methods to be used when a numpy array comes first
    __array_priority__ = 1

    input_shape = None
    output_shape = None
    differentiable = False
    affine = False
    quadratic = False

    def __init__(self):
        self._setdiff()

    def _expression(self):
        return self.__class__.__name__

    def _to_cvxpy(self):
        pass

    def _extend_variable(self, n_var, axis, pos):
        pass

    def __str__(self):
        description = {'expression': self._expression(),
                       'input shape': self.input_shape,
                       'output shape': self.output_shape,
                       'affine': self.is_affine,
                       'quadratic': self.is_quadratic
                       }
        return str(description)

    def __neg__(self):
        return NegFunction(self)

    def __add__(self, other):
        if isinstance(other, AbstractFunction):
            return SumFunction(self, other)
        elif isinstance(other, (int, float, np.ndarray)):
            return ConstantSumFunction(other, self)
        else:
            raise TypeError

    def __radd__(self, other: Union[int, float, np.ndarray]):
        if isinstance(other, (int, float, np.ndarray)):
            return ConstantSumFunction(other, self)
        else:
            raise TypeError

    def __iadd__(self, other):
        if isinstance(other, AbstractFunction):
            return SumFunction(self, other)
        elif isinstance(other, (int, float, np.ndarray)):
            return ConstantSumFunction(other, self)
        else:
            raise TypeError

    def __sub__(self, other):
        if isinstance(other, AbstractFunction):
            return SumFunction(self, -other)
        elif isinstance(other, (int, float, np.ndarray)):
            return ConstantSumFunction(-other, self)
        else:
            raise TypeError

    def __rsub__(self, other):
        if isinstance(other, AbstractFunction):
            return SumFunction(-self, other)
        elif isinstance(other, (int, float, np.ndarray)):
            return ConstantSumFunction(other, -self)
        else:
            raise TypeError

    def __isub__(self, other):
        if isinstance(other, AbstractFunction):
            return SumFunction(self, -other)
        elif isinstance(other, (int, float, np.ndarray)):
            return ConstantSumFunction(-other, self)
        else:
            raise TypeError

    def __mul__(self, other):
        if isinstance(other, AbstractFunction):
            return MulFunction(self, other)
        elif isinstance(other, (int, float)):
            return ScalarMulFunction(other, self)
        else:
            raise TypeError

    def __rmul__(self, other: Union[int, float]):
        if isinstance(other, (int, float)):
            return ScalarMulFunction(other, self)
        else:
            raise NotImplementedError

    def __imul__(self, other):
        if isinstance(other, AbstractFunction):
            return MulFunction(self, other)
        elif isinstance(other, (int, float)):
            return ScalarMulFunction(other, self)
        else:
            raise TypeError
   
    def __truediv__(self, other):
        if isinstance(other, AbstractFunction):
            from .power import Power
            return MulFunction(self, Power(other, -1))
        elif isinstance(other, (int, float)):
            return ScalarMulFunction(1.0/other, self)
        else:
            raise TypeError

    def __rdiv__(self, other: Union[int, float]):
        if isinstance(other, (int, float)):
            return ScalarMulFunction(1.0/other, self)
        else:
            raise NotImplementedError

    def __idiv__(self, other):
        if isinstance(other, AbstractFunction):
            from .power import Power
            return MulFunction(self, Power(other, -1))
        elif isinstance(other, (int, float)):
            return ScalarMulFunction(1.0/other, self)
        else:
            raise TypeError

    def __pow__(self, other):
        if isinstance(other, (int, float)):
            from .power import Power
            return Power(self, float(other))
        else:
            raise TypeError

    def __ipow__(self, other):
        if isinstance(other, (int, float)):
            from .power import Power
            return Power(self, float(other))
        else:
            raise TypeError

    def __matmul__(self, other):
        if isinstance(other, AbstractFunction):
            return MatMulFunction(self, other)
        elif isinstance(other, (np.ndarray)):
            if other.shape == self.output_shape:
                return ConstantMatMulFunction(other, self)
            else:
                raise NotImplementedError
        else:
            raise TypeError

    def __rmatmul__(self, other: np.ndarray):
        if isinstance(other, (np.ndarray)):
            return ConstantMatMulFunction(other, self)
        else:
            raise TypeError

    def __imatmul__(self, other: np.ndarray):
        return self.__matmul__(other)

    def __to_constraint(self, other, sign):
        if not isinstance(other, (AbstractFunction, np.ndarray, float, int)):
            raise TypeError

        if isinstance(other, AbstractFunction):
            if (self.input_shape != other.input_shape) or (self.output_shape != other.output_shape):
                raise ValueError("Incompatible shapes")
            from ..constraints.constraints import Constraint
            return Constraint(self - other, sign=sign)

        if isinstance(other, np.ndarray):
            if other.shape != self.output_shape:
                raise ValueError("Incompatible shapes")
            from ..constraints.constraints import Constraint
            return Constraint(self - other, sign=sign)

        if isinstance(other, (int, float)):
            if self.output_shape != (1, 1):
                const = other * np.ones(self.output_shape)
            else:
                const = other
            from ..constraints.constraints import Constraint
            return Constraint(self - const, sign=sign)

    def __eq__(self, other):
        return self.__to_constraint(other, "==")

    def __le__(self, other):
        return self.__to_constraint(other, "<=")

    def __ge__(self, other):
        return self.__to_constraint(other, ">=")

    @property
    def is_differentiable(self):
        return self.differentiable

    @property
    def is_affine(self):
        return self.affine

    @property
    def is_quadratic(self):
        return self.quadratic

    def get_parameters(self):
        raise NotImplementedError

    def _setdiff(self):
        self._subgradient = ag_grad(self.eval)
        self._jacobian = ag_jacobian(self.eval)
        self._hessian = ag_hessian(self.eval)

    @check_input
    def eval(self, x: np.ndarray) -> np.ndarray:
        """Evaluate the function at a point x

        Args:
            x: input point
        """
        pass

    @check_input
    def jacobian(self, x: np.ndarray, **kwargs) -> np.ndarray:
        """Evaluate the jacobian of the the function at a point x

        Args:
            x: input point
        """
        warnings.simplefilter("error", category=RuntimeWarning)
        warnings.simplefilter("ignore", category=UserWarning)  # TODO: check
        try:
            jac = np.squeeze(self._jacobian(x))
        except RuntimeWarning:
            try:
                jac = self._alternative_jacobian(x)
            except NotImplementedError:
                raise NotImplementedError("No jacobian can be computed")

        if jac.shape == ():  # in case of scalar
            jac = jac.reshape(1, 1)
        if len(jac.shape) == 1:
            jac = jac.reshape(1, -1)
        # np.nan_to_num(jac, copy=False)
        return jac

    @check_input
    def _alternative_jacobian(self, x: np.ndarray, **kwargs) -> np.ndarray:
        """Evaluate the jacobian of the the function at a point x

        Args:
            x: input point
        """
        raise NotImplementedError

    @check_input
    def hessian(self, x: np.ndarray, **kwargs) -> np.ndarray:
        """Evaluate the hessian of the the function at a point x

        Args:
            x: input point
        """
        warnings.simplefilter("error", category=RuntimeWarning)
        warnings.simplefilter("ignore", category=UserWarning)  # TODO: check
        try:
            hess = np.squeeze(self._hessian(x))
        except RuntimeWarning:
            try:
                hess = np.squeeze(self._alternative_hessian(x))
            except NotImplementedError:
                # TODO risolvere caso in cui autograd non riesce: np.true_divide(inf, inf)
                raise NotImplementedError("No hessian can be computed")

        if hess.shape == ():  # in case of scalar
            hess = hess.reshape(1, 1)
        # np.nan_to_num(hess, copy=False)
        return hess

    @check_input
    def _alternative_hessian(self, x: np.ndarray, **kwargs) -> np.ndarray:
        """Evaluate the hessian of the the function at a point x

        Args:
            x: input point
        """
        raise NotImplementedError

    @check_input
    def subgradient(self, x: np.ndarray, **kwargs) -> np.ndarray:
        """Evaluate the subgradient of the function at a point x

        Args:
            x: input point

        Raises:
            ValueError: subgradient is defined only for functions with scalar output 
        """
        warnings.simplefilter("error", category=RuntimeWarning)
        warnings.simplefilter("ignore", category=UserWarning)  # TODO: check
        if not self.output_shape == (1, 1):
            raise ValueError("Undefined subgradient")
        else:
            try:
                subg = self._alternative_subgradient(x).reshape(self.input_shape)
            except NotImplementedError:
                try:
                    subg = self._subgradient(x).reshape(self.input_shape)
                    # subg = self._alternative_subgradient(x).reshape(self.input_shape)
                except RuntimeWarning:
                    raise NotImplementedError("No subgradient can be computed")
            # subg = self.jacobian(x).reshape(self.input_shape)
            # np.nan_to_num(subg, copy=False)
            return subg

    @check_input
    def _alternative_subgradient(self, x: np.ndarray, **kwargs) -> np.ndarray:
        """Evaluate the hessian of the the function at a point x

        Args:
            x: input point
        """
        return self._alternative_jacobian(x).reshape(self.input_shape)


class ConstantSumFunction(AbstractFunction):
    """Add a constant to a AbstractFunction

    Args:
        constant (numpy.ndarray): constant to add
        fn (AbstractFunction): function to add a constant to

    Raises:
        ValueError: The shape of the constant must be equal to the output shape of the function
    """

    def __init__(self, constant: Union[int, float, np.ndarray], fn: AbstractFunction):
        self.constant = constant
        if isinstance(constant, np.ndarray):
            if fn.output_shape != constant.shape:
                raise ValueError("Different shapes")
        self.fn = fn
        self.input_shape = fn.input_shape
        self.output_shape = fn.output_shape
        self.differentiable = fn.is_differentiable
        if fn.is_affine:
            self.affine = True
        if fn.is_quadratic:
            self.quadratic = True

        super().__init__()

    def _expression(self):
        return str(self.constant) + " + " + self.fn._expression()

    @check_input
    def eval(self, x):
        return self.fn.eval(x) + self.constant

    def _to_cvxpy(self):
        return self.constant + self.fn._to_cvxpy()
    
    def _extend_variable(self, n_var, axis, pos):
        return self.fn._extend_variable(n_var, axis, pos) + self.constant

    @check_input
    def jacobian(self, x):
        return self.fn.jacobian(x)

    @check_input
    def hessian(self, x):
        return self.fn.hessian(x)

    @check_input
    def subgradient(self, x):
        return self.fn.subgradient(x)


class SumFunction(AbstractFunction):
    """Sum two AbstractFunction objects

    Args:
        f1 (AbstractFunction): first function
        f2 (AbstractFunction): second function

    Raises:
        ValueError: The input/output shapes of the two functions must be the same
    """

    def __init__(self, f1: AbstractFunction, f2: AbstractFunction):
        if f1.input_shape != f2.input_shape or f1.output_shape != f2.output_shape:
            raise ValueError("Different input/output shapes")
        self.f1 = f1
        self.f2 = f2
        self.input_shape = f1.input_shape
        self.output_shape = f1.output_shape
        if f1.is_differentiable and f2.is_differentiable:
            self.differentiable = True
        if f1.is_affine and f2.is_affine:
            self.affine = True
        if (f1.is_quadratic and f2.is_quadratic) or (f1.is_quadratic and f2.is_affine) or (f1.is_affine and f2.is_quadratic):
            self.quadratic = True

        super().__init__()

    def _expression(self):
        return self.f1._expression() + " + " + self.f2._expression()

    def _to_cvxpy(self):
        return self.f1._to_cvxpy() + self.f2._to_cvxpy()
    
    def _extend_variable(self, n_var, axis, pos):
        return self.f1._extend_variable(n_var, axis, pos) + self.f2._extend_variable(n_var, axis, pos)

    @check_input
    def eval(self, x):
        return self.f1.eval(x) + self.f2.eval(x)

    @check_input
    def jacobian(self, x):
        return self.f1.jacobian(x) + self.f2.jacobian(x)

    @check_input
    def hessian(self, x):
        return self.f1.hessian(x) + self.f2.hessian(x)

    @check_input
    def subgradient(self, x):
        return self.f1.subgradient(x) + self.f2.subgradient(x)


class NegFunction(AbstractFunction):
    """Multiplies a AbstractFunction by -1

    Args:
        fn (AbstractFunction): input function function
    """

    def __init__(self, fn: AbstractFunction):
        self.fn = fn
        self.input_shape = fn.input_shape
        self.output_shape = fn.output_shape
        self.differentiable = fn.is_differentiable
        if fn.is_affine:
            self.affine = True

        if fn.is_quadratic:
            self.quadratic = True

        super().__init__()

    def _expression(self):
        return " - " + self.fn._expression()

    def _to_cvxpy(self):
        return -self.fn._to_cvxpy()
    
    def _extend_variable(self, n_var, axis, pos):
        return -self.fn._extend_variable(n_var, axis, pos)

    @check_input
    def eval(self, x):
        return -self.fn.eval(x)

    @check_input
    def jacobian(self, x):
        return -self.fn.jacobian(x)

    @check_input
    def hessian(self, x):
        return -self.fn.hessian(x)

    @check_input
    def subgradient(self, x):
        return -self.fn.subgradient(x)


class MulFunction(AbstractFunction):
    def __init__(self, f1: AbstractFunction, f2: AbstractFunction):
        """Multiplies two functions (with scalar outputs)

        Args:
            f1 (AbstractFunction): first function
            f2 (AbstractFunction): second function

        Raises:
            ValueError: input functions must have the same input shapes
            ValueError: at least the first input function must have scalar output
            NotImplementedError: only differentiable input functions are currently supported
        """
        if f1.input_shape != f2.input_shape:
            raise ValueError("Different input shapes")
        if not(f1.output_shape == (1, 1)):
            raise ValueError(
                "At least the first input function must have scalar output")
        self.f1 = f1
        self.f2 = f2
        self.input_shape = f1.input_shape
        self.output_shape = f2.output_shape
        if f1.is_differentiable and f2.is_differentiable:
            self.differentiable = True

        super().__init__()

    def _expression(self):
        return self.f1._expression() + " * " + self.f2._expression()

    @check_input
    def eval(self, x):
        return self.f1.eval(x) * self.f2.eval(x)

    def _extend_variable(self, n_var, axis, pos):
        return self.f1._extend_variable(n_var, axis, pos) * self.f2._extend_variable(n_var, axis, pos)

    def _to_cvxpy(self):
        return self.f1._to_cvxpy() * self.f2._to_cvxpy()
    

class ScalarMulFunction(AbstractFunction):
    def __init__(self, scalar: Union[int, float], fn: AbstractFunction):
        """Multiply a function by a scalar

        Args:
            scalar (float or int): scalar
            fn (AbstractFunction): input function
        Raises:
            ValueError: the scalar must be a float (or an int)
        """
        if not isinstance(scalar, (int, float)):
            raise ValueError(
                "Multiplication by scalars requires the scalar to be float or int")
        self.scalar = scalar
        self.fn = fn
        self.input_shape = fn.input_shape
        self.output_shape = fn.output_shape
        if fn.is_differentiable:
            self.differentiable = True
        if fn.is_affine:
            self.affine = True
        if fn.is_quadratic:
            self.quadratic = True

        super().__init__()

    def _expression(self):
        return str(self.scalar) + " * " + self.fn._expression()

    def _to_cvxpy(self):
        return self.scalar * self.fn._to_cvxpy()

    @check_input
    def eval(self, x):
        return self.scalar * self.fn.eval(x)
    
    def _extend_variable(self, n_var, axis, pos):
        return self.scalar * self.fn._extend_variable(n_var, axis, pos)

    @check_input
    def jacobian(self, x):
        return self.scalar * self.fn.jacobian(x)

    @check_input
    def hessian(self, x):
        return self.scalar * self.fn.hessian(x)

    @check_input
    def subgradient(self, x):
        return self.scalar * self.fn.subgradient(x)


class MatMulFunction(AbstractFunction):
    def __init__(self, f1: AbstractFunction, f2: AbstractFunction):
        """Dot product of two function

        :math:`f(x) = \\langle f_1(x), f_2(x)\\rangle=\\f_1(x)^\\top f_2(x)

        with :math:`f_1,f_2:\\mathbb{R}^m\\to\\mathbb{R}^n`

        Args:
            f1 (AbstractFunction): first function (output shape (n, 1))
            f2 (AbstractFunction): second function (output shape (n, 1))

        Raises:
            ValueError: input functions must have the same input shapes
            NotImplementedError: only differentiable input functions are currently supported
        """
        if not (f1.input_shape == f2.input_shape and f1.output_shape == f2.output_shape):
            raise ValueError("Different input/output shapes")

        self.f1 = f1
        self.f2 = f2
        self.input_shape = f1.input_shape
        self.output_shape = (1, 1)

        if f1.output_shape[1] != 1:
            raise ValueError("Unsupported input functions output shape. Must be (n, 1)")

        if f1.is_differentiable and f2.is_differentiable:
            self.differentiable = True

        super().__init__()

    def _expression(self):
        return self.f1._expression() + " @ " + self.f2._expression()

    @check_input
    def eval(self, x):
        return self.f1.eval(x).transpose() @ self.f2.eval(x)

    def _extend_variable(self, n_var, axis, pos):
        return MatMulFunction(self.f1._extend_variable(n_var, axis, pos), self.f2._extend_variable(n_var, axis, pos))

    def _to_cvxpy(self):
        return self.f1._to_cvxpy().transpose() @ self.f2._to_cvxpy()


class ConstantMatMulFunction(AbstractFunction):
    def __init__(self, constant: np.ndarray, fn: AbstractFunction):
        """Matrix/Dot product of a matrix/vector and a function

        :math:`f(x) = \\langle A, g(x)\\rangle=\\A^\\top g(x)

        with :math:`A:\\mathbb{R}^{n\\times m}` and :math:`g:\\mathbb{R}^d\\to\\mathbb{R}^n`

        Args:
            constant (numpy.ndarray): constant vector (shape (n, m))
            fn (AbstractFunction): function (output shape (n, 1))

        Raises:
            ValueError: input functions must have the same input shapes
            NotImplementedError: only differentiable input functions are currently supported
        """
        if not isinstance(constant, (np.ndarray)):
            raise ValueError(
                "Multiplication by constant vector requires the vector to be a numpy.ndarray")
        if fn.output_shape[0] != constant.shape[0]:
            raise ValueError(
                "Incompatible dimensions: function ouput is {} and given matrix is {}".format(
                    fn.output_shape, constant.shape))

        self.constant = constant
        self.fn = fn
        self.input_shape = fn.input_shape
        self.output_shape = (self.constant.shape[1], self.fn.output_shape[1])

        if fn.output_shape[1] != 1:
            raise ValueError("Unsupported function output shape. Must be (n, 1)")

        if fn.is_differentiable:
            self.differentiable = True

        if fn.is_affine:
            self.affine = True

        super().__init__()

    def _expression(self):
        return str(self.constant()) + " @ " + self.fn._expression()

    @check_input
    def eval(self, x):
        return self.constant.transpose() @ self.fn.eval(x)

    def _extend_variable(self, n_var, axis, pos):
        return ConstantMatMulFunction(self.constant, self.fn._extend_variable(n_var, axis, pos))

    def _to_cvxpy(self):
        return self.constant.transpose() @ self.fn._to_cvxpy()
