import numpy as np
import scipy.sparse as sp
import warnings
from .problem import Problem
from ..functions import AffineForm, QuadraticForm
from ..functions.affine_form import aggregate_affine_form
from ..constraints import Constraint
from ..utils.utilities import check_symmetric, is_semi_pos_def
from .utilities import check_affine_constraints


class QuadraticProblem(Problem):
    """Solve a Quadratic programming problem defined as:

    .. math::

        \\text{minimize } & x^\\top P x + q^\\top x + r 

        \\text{subject to } & G x \\leq h

                            & A x = b

    Quadratic problems are currently solved by using CVXOPT or OSQP https://osqp.org. 

    Args:
        objective_function (QuadraticForm): objective function
        constraints (list, optional): list of constraints. Defaults to None.
        is_pos_def (bool): True if P is (semi)positive definite. Defaults to True.
    """

    def __new__(cls, objective_function: QuadraticForm = None, constraints: list = None, is_pos_def: bool = True):
        instance = object.__new__(cls)
        return instance

    def __init__(self, objective_function: QuadraticForm, constraints: list = None, is_pos_def: bool = True):
        self.objective_function = None
        self.constraints = []
        self.input_shape = None
        self.output_shape = None
        self.is_pos_def = is_pos_def
        self.set_objective_function(objective_function)
        if not check_affine_constraints(constraints):
            raise ValueError("Constraints must be affine")
        self.aggregated_constraints = []
        if constraints is not None:
            self.set_constraints(constraints)
        # TODO: check init is done twice if instantiated via Problem

    def set_objective_function(self, objective_function: QuadraticForm):
        """set the objective function

        Args:
            objective_function (QuadraticForm): objective function

        Raises:
            TypeError: Objective function must be a QuadraticForm
        """
        if not isinstance(objective_function, QuadraticForm):
            raise TypeError("Objective function must be a QuadraticForm")
        P, _, _ = objective_function.get_parameters()
        if not self.is_pos_def:
            if not is_semi_pos_def(P):
                warnings.warn("Objective function is not convex.")
        self.objective_function = objective_function
        self.input_shape = objective_function.input_shape
        self.output_shape = objective_function.output_shape

    def set_constraints(self, constraints):
        """Set the constraints

        Args:
            constraints (list): list of constraints

        Raises:
            TypeError: a list of affine Constraints must be provided
        """
        if not isinstance(constraints, list):
            constraints = [constraints]
        for constraint in constraints:
            super().add_constraint(constraint)

        self.constraints_map = {}
        equality_idx_counter = 0
        inequality_idx_counter = 0
        equalities = []
        inequalities = []
        for idx, constraint in enumerate(self.constraints):
            if constraint.is_affine:
                if constraint.is_equality:
                    equalities.append(constraint.function)
                    self.constraints_map[idx] = {'type': 'affine_equality', 'indices': list(
                        range(equality_idx_counter, equality_idx_counter+constraint.output_shape[0]))}
                    equality_idx_counter += constraint.output_shape[0]
                elif constraint.is_inequality:
                    inequalities.append(constraint.function)
                    self.constraints_map[idx] = {'type': 'affine_inequality', 'indices': list(
                        range(inequality_idx_counter, inequality_idx_counter+constraint.output_shape[0]))}
                    inequality_idx_counter += constraint.output_shape[0]
            else:
                raise TypeError("a list of affine Constraints must be provided")

        for key, item in self.constraints_map.items():
            if item['type'] == 'affine_inequality':
                item['indices'] = list(np.array(item['indices']) + equality_idx_counter)

        if len(equalities) != 0:
            aggregated_equality = aggregate_affine_form(equalities)
            self.aggregated_constraints.append(aggregated_equality == 0)

        if len(inequalities) != 0:
            aggregated_inequality = aggregate_affine_form(inequalities)
            self.aggregated_constraints.append(aggregated_inequality <= 0)

    # def add_constraint(self, constraint):
    #     """A a new (affine) constraint

    #     Args:
    #         constraint (Constraint): affine constraint

    #     Raises:
    #         TypeError: an affine Constraints must be provided
    #     """
    #     if not isinstance(constraint, Constraint):
    #         raise TypeError("an affine Constraints must be provided")
    #     if constraint.is_affine:
    #         if constraint.is_equality:
    #             if len(self.constraints) == 0:
    #                 self.constraints.append(constraint)
    #             else:
    #                 equality_found = False
    #                 for const in self.constraints:
    #                     if const.is_equality:
    #                         equality_found = True
    #                         aggregated_equality = aggregate_affine_form([const.function, constraint.function])
    #                         self.constraints.remove(const)
    #                         self.constraints.append(aggregated_equality == 0)

    #                 if not equality_found:
    #                     self.constraints.append(constraint)
    #         elif constraint.is_inequality:
    #             if len(self.constraints) == 0:
    #                 self.constraints.append(constraint)
    #             else:
    #                 inequality_found = False
    #                 for const in self.constraints:
    #                     if const.is_inequality:
    #                         inequality_found = True
    #                         aggregated_inequality = aggregate_affine_form([const.function, constraint.function])
    #                         self.constraints.remove(const)
    #                         self.constraints.append(aggregated_inequality <= 0)

    #                 if not inequality_found:
    #                     self.constraints.append(constraint)
    #     else:
    #         raise TypeError("an affine Constraints must be provided")

    def __get_osqp_objective_parameters(self):
        """return the objective parameter fo osqp solver
        """
        P, q, _ = self.objective_function.get_parameters()
        if isinstance(P, np.ndarray):
            if not check_symmetric(P):
                raise ValueError("Matrix P must be symmetric.")
            if not is_semi_pos_def(P):
                raise ValueError("Nonconvex objective function.")
            P = sp.csc_matrix(2*P)
        return P, q.flatten()

    def __get_osqp_constraints_parameters(self):
        """return the constraints parameter fo osqp solver
        """
        A = None
        b = None
        G = None
        h = None
        if len(self.aggregated_constraints) != 0:
            for const in self.aggregated_constraints:
                if const.is_equality:
                    A, b = const.get_parameters()
                if const.is_inequality:
                    G, h = const.get_parameters()

            if A is not None:
                A_qp = A
                l_qp = -b
                u_qp = -b

                if G is not None:
                    A_qp = np.hstack([A, G])
                    l_qp = np.vstack([-b, -np.inf * np.ones((len(h), 1))])
                    u_qp = np.vstack([-b, -h])
            else:
                A_qp = G
                l_qp = -np.inf * np.ones(len(h))
                u_qp = -h

            A_osqp = sp.csc_matrix(A_qp.transpose())
            l_osqp = l_qp.flatten()
            u_osqp = u_qp.flatten()

            return A_osqp, l_osqp, u_osqp
        else:
            return None, None, None

    def __solve_osqp(self, initial_value: np.ndarray = None, return_only_solution: bool = True):
        """return the problem through osqp
        """
        try:
            from osqp import OSQP
        except ImportError:
            raise ImportError("OSQP is not installed")

        P, q = self.__get_osqp_objective_parameters()
        A, l, u = self.__get_osqp_constraints_parameters()
        osqp_pb = OSQP()
        osqp_pb.setup(P=P, q=q, A=A, l=l, u=u, verbose=False)
        osqp_pb.update_settings(eps_abs=1e-7, eps_rel=1e-7, max_iter=10000)

        if initial_value is not None:
            osqp_pb.warm_start(x=initial_value)
        res = osqp_pb.solve()
        if res.info.status != 'solved':
            warnings.warn("OSQP exited with status {}".format(res.info.status))

        solution = res.x.reshape(self.input_shape)
        if not return_only_solution:
            dual_variables = {}
            for idx, _ in enumerate(self.constraints):
                dual_variables[idx] = res.y[self.constraints_map[idx]['indices']].reshape(-1, 1)

            output_dict = {
                'solution': solution,
                'status': res.info.status,
                'dual_variables': dual_variables,
            }
            return output_dict
        else:
            return solution

    def __get_cvxopt_objective_parameters(self):
        """return the objective parameter fo osqp solver
        """
        P, q, _ = self.objective_function.get_parameters()
        if isinstance(P, np.ndarray):
            if not is_semi_pos_def(P):
                raise ValueError("Nonconvex objective function.")
        return 2*P, q

    def __get_cvxopt_constraints_parameters(self):
        """return the constraints parameter fo osqp solver
        """
        A = None
        b = None
        G = None
        h = None
        if len(self.aggregated_constraints) != 0:
            for const in self.aggregated_constraints:
                if const.is_equality:
                    A, b = const.get_parameters()
                    A = A.transpose()
                    b = -b
                if const.is_inequality:
                    G, h = const.get_parameters()
                    G = G.transpose()
                    h = -h
        return A, b, G, h

    def __solve_cvxopt(self, initial_value: np.ndarray = None, return_only_solution: bool = True):
        """return the problem through osqp
        """
        try:
            import cvxopt
        except ImportError:
            raise ImportError("CVXOPT is required")
        cvxopt.solvers.options['show_progress'] = False
        cvxopt.solvers.options['maxiters'] = 10000

        P, q = self.__get_cvxopt_objective_parameters()
        A, b, G, h = self.__get_cvxopt_constraints_parameters()
        if initial_value is not None:
            initvals = {'x': initial_value}
        else:
            initvals = None
        if A is not None:
            A = cvxopt.matrix(A)
            b = cvxopt.matrix(b)
        if G is not None:
            G = cvxopt.matrix(G)
            h = cvxopt.matrix(h)

        sol = cvxopt.solvers.qp(P=cvxopt.matrix(P),
                                q=cvxopt.matrix(q),
                                G=G,
                                h=h,
                                A=A,
                                b=b,
                                initvals=initvals)

        solution = np.array(sol['x']).reshape(self.input_shape)
        if not return_only_solution:
            dual_variables = {}
            for idx, constr in enumerate(self.constraints):
                if constr.is_inequality:
                    dual_variables[idx] = np.array(sol['z'])[self.constraints_map[idx]['indices']]
                elif constr.is_equality:
                    dual_variables[idx] = np.array(sol['y'])[self.constraints_map[idx]['indices']]

            if sol['status'] == 'optimal':
                status = 'solved'
            else:
                status = sol['status']

            output_dict = {
                'solution': solution,
                'status': status,
                'dual_variables': dual_variables,
            }
            return output_dict
        else:
            return solution

    def solve(self, initial_value: np.ndarray = None, solver='osqp', return_only_solution: bool = True):
        """Solve the problem

        Args:
            initial_value (numpy.ndarray), optional): Initial value for warm start. Defaults to None.
            solver (str, optional): Solver to use ('osqp' or 'cvxopt'). Defaults to 'osqp'.

        Raises:
            ValueError: Unsupported solver, only 'osqp' and 'cvxopt' are currently supported

        Returns:
            numpy.ndarray: solution
        """
        P, _ = self.__get_cvxopt_objective_parameters()
        if solver == 'osqp':
            if not check_symmetric(P):
                solver = 'cvxopt'
                warnings.warn("OSQP solver requires the matrix of the quadratic function to be symmetric. Solving with cvxopt.")
        if solver == 'osqp':
            return self.__solve_osqp(initial_value=initial_value, return_only_solution=return_only_solution)
        elif solver == 'cvxopt':
            try:
                return self.__solve_cvxopt(initial_value=initial_value, return_only_solution=return_only_solution)
            except:
                return super().solve(return_only_solution=return_only_solution)
        else:
            raise ValueError("Unsupported solver")
