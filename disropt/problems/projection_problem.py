import numpy as np
from .problem import Problem
from .quadratic_problem import QuadraticProblem


class ProjectionProblem(Problem):
    """Computes the projection of a point onto some constraints, i.e., it solves

    .. math::

        \\text{minimize } & \\frac{1}{2}\\| x - p \\|^2 

        \\text{subject to } & f_k(x)\\leq 0, \\, k=1,...


    Args:
        constraints_list (list): list of constraints
        point (numpy.ndarray): point :math:`p` to project
    """

    def __init__(self, constraints_list, point):
        from ..functions import Variable, QuadraticForm
        x = Variable(point.shape[0])
        objective_function = 0.5 * (x - point) @ (x - point)
        self.point = point

        super(ProjectionProblem, self).__init__(objective_function, constraints_list)

        self.only_affine_constraints = False
        if len(self.nonlinear_constraints) == 0:
            self.only_affine_constraints = True

    def solve(self):
        """solve the problem

        Returns:
            numpy.ndarray: solution
        """
        satisfied = True
        if self.affine_equality:
            satisfied &= self.affine_equality.eval(self.point)
        if self.affine_inequality:
            satisfied &= self.affine_inequality.eval(self.point)
        for constr in self.nonlinear_constraints:
            satisfied &= constr.eval(self.point)

        if satisfied:
            return self.point
        else:
            if self.only_affine_constraints:
                # if non fat matrix, projection is explicit
                A_eq = None
                A_ineq = None
                if self.affine_equality:
                    A_eq, b_eq = self.affine_equality.get_parameters()
                if self.affine_inequality:
                    A_ineq, b_ineq = self.affine_inequality.get_parameters()
                if (A_eq is not None) and (A_ineq is not None):
                    A = np.hstack([A_eq, A_ineq])
                    b = np.vstack([b_eq, b_ineq])
                elif A_eq is not None:
                    A = A_eq
                    b = b_eq
                else:
                    A = A_ineq
                    b = b_ineq
                if A.shape[0] >= A.shape[1]:
                    projected_pt = self.point - A @ np.linalg.inv(A.transpose() @ A) @ (A.transpose() @ self.point + b)
                    return projected_pt
                else:
                    from .quadratic_problem import QuadraticProblem
                    qp = QuadraticProblem(self.objective_function, self.constraints, is_pos_def=True)
                    return qp.solve()
            else:
                return super().solve()
