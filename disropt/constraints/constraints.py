import numpy as np
from ..functions.utilities import check_input


class AbstractConstraint:
    """Abstract class for expressing constraints
    """
    def eval(self):
        pass


class Constraint(AbstractConstraint):
    """Constraint build from a AbstractFunction object. Constraints are represented in the canonical forms :math:`f(x)=0` and :math:`f(x)\leq 0`.

    Args:
        fn (AbstractFunction): constraint function 
        sign (bool): type of constraint: "==", "<=" or ">="

    Attributes:
        fn (AbstractFunction): constraint function 
        sign (bool): type of constraint: "==", "<=" or ">="
        input_shape (tuple): input space dimensions
        output_shape (tuple): output space dimensions
    """

    def __init__(self, fn, sign: str = "=="):
        self.input_shape = fn.input_shape
        self.output_shape = fn.output_shape
        self.sign = sign
        self.fn = fn

        if self.sign == ">=":
            self.sign = "<="
            self.fn = -fn
    
    def _to_cvxpy(self):
        if self.is_equality:
            return self.fn._to_cvxpy() == 0
        else:
            return self.fn._to_cvxpy() <= 0
            
    @property
    def is_equality(self):
        return self.sign == "=="
    
    @property
    def is_inequality(self):
        return self.sign == "<="

    @property
    def is_affine(self):
        """Return true if the function is affine.

        Returns:
            bool: true if the function is affine
        """
        return self.fn.is_affine

    @property
    def is_quadratic(self):
        """Return true if the function is affine.

        Returns:
            bool: true if the function is affine
        """
        return self.fn.is_quadratic

    @property
    def function(self):
        return self.fn

    def get_parameters(self):
        """Return the parameters of the function if it is affine or quadratic

        Returns:
            tuple: A, b for affine constraints, P, q, r for quadratic
        """
        return self.fn.get_parameters()

    @check_input
    def eval(self, x: np.ndarray) -> bool:
        """Evaluate the constraint function at a point x

        Args:
            x: input point
        """
        if self.is_equality:
            return np.allclose(self.fn.eval(x), np.zeros(self.output_shape))
        elif self.is_inequality:
            if np.allclose(self.fn.eval(x), np.zeros(self.output_shape)):
                return True
            return np.less_equal(self.fn.eval(x), np.zeros(self.output_shape)).all()

    @check_input
    def projection(self, x: np.ndarray) -> np.ndarray:
        """Compute the projection of a point onto the set defined by the constraint. The constraint should be convex.
        
        Args:
            x : point to be projected
        
        Returns:
            numpy.ndarray: projected point
        """
        satisfied = self.eval(x)
        if satisfied:
            return x
        else:
            if self.is_affine:
                A, b = self.get_parameters()
                if A.shape[0] >= A.shape[1]:
                    projected_pt = x - A @ np.linalg.inv(A.transpose() @ A) @ (A.transpose() @ x + b)
                    return projected_pt
                else:
                    from ..problems import Problem
                    from ..functions import Variable, QuadraticForm
                    v = Variable(self.input_shape[0])
                    P = np.eye(self.input_shape[0])
                    obj = 0.5 * QuadraticForm(v - x, P)
                    constraint = self
                    pb = Problem(obj, constraint)
                    projected_pt = pb.solve()
                    return projected_pt
            else:
                from ..problems import Problem
                from ..functions import Variable, QuadraticForm
                v = Variable(self.input_shape[0])
                P = np.eye(self.input_shape[0])
                obj = 0.5 * QuadraticForm(v - x, P)
                constraint = self
                pb = Problem(obj, constraint)
                projected_pt = pb.solve()
                return projected_pt

