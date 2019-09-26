import numpy as np
from .problem import Problem
from ..functions import AffineForm
from ..functions.affine_form import aggregate_affine_form
from ..constraints import Constraint, AbstractSet
from .utilities import check_affine_constraints


class LinearProblem(Problem):
    """Solve a Linear programming problem defined as:

    .. math::

        \\text{minimize } & c^\\top x

        \\text{subject to } & G x \\leq h

                            & A x = b
    """

    def __init__(self, objective_function: AffineForm, constraints: list = None):
        """[summary]

        Args:
            objective_function (AffineForm): [description]
            constraints (list, optional): [description]. Defaults to None.
        """
        self.objective_function = None
        self.constraints = []
        self.input_shape = None
        self.output_shape = None
        self.set_objective_function(objective_function)
        if not check_affine_constraints(constraints):
            raise ValueError("Constraints must be affine")
        self.__set_constraints(constraints)

    def set_objective_function(self, objective_function: AffineForm):
        """set the objective function

        Args:
            objective_function (AffineForm): objective function

        Raises:
            TypeError: Objective function must be a AffineForm with output_shape=(1,1)
        """
        if not isinstance(objective_function, AffineForm):
            raise TypeError("Objective function must be a AffineForm")
        if not objective_function.output_shape == (1,1):
            raise TypeError("Objective function must be a AffineForm with output_shape=(1,1)")

        self.objective_function = objective_function
        self.input_shape = objective_function.input_shape
        self.output_shape = objective_function.output_shape

    def __set_constraints(self, constraints):
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

        self.aggregated_constraints = []
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
    #     if not isinstance(constraint, (Constraint, AbstractSet)):
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

    def __get_cvxopt_objective_parameters(self):
        """Return the objective parameter required by cvxopt

        Returns:
            numpy.ndarray
        """
        c,  _ = self.objective_function.get_parameters()
        return c

    def __get_cvxopt_constraints_parameters(self):
        """Return the constraints parameters required by cvxopt

        Returns:
            tuple: (A, b, G, h)
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

        return G, h, A, b

    def __solve_cvxopt(self, initial_value: np.ndarray = None, solver='glpk', return_only_solution: bool = True):
        """Solve the problem through cvxopt interface

        Args:
            initial_value (np.ndarray, optional): starting point for warm start (if possible). Defaults to None.
            solver (str, optional): ('glpk' or 'cvxopt'). Defaults to 'glpk'.

        Raises:
            ValueError: solver must be 'glpk' or 'cvxopt'
            ValueError: optimal solution not found

        Returns:
            numpy.ndarray: solution
        """
        try:
            import cvxopt
        except ImportError:
            raise ImportError("CVXOPT is required")
        cvxopt.solvers.options['show_progress'] = False
        cvxopt.solvers.options['maxiters'] = 10000
        if solver == 'glpk':
            try:
                import cvxopt.glpk
                solver = 'glpk'
                cvxopt.solvers.options['glpk'] = {'msg_lev': 'GLP_MSG_OFF'}
            except ImportError:
                print("GLPK solver not found. Trying with cvxopt...")
                solver = None
        elif solver == 'cvxopt':
            solver = None
        else:
            raise ValueError("Unsupported solver")

        c = self.__get_cvxopt_objective_parameters()
        G, h, A, b = self.__get_cvxopt_constraints_parameters()
        args = [cvxopt.matrix(c)]
        if G is not None:
            G = cvxopt.matrix(G)
            h = cvxopt.matrix(h)
        else:
            G = cvxopt.matrix(np.ones(c.shape).transpose())
            h = cvxopt.matrix(np.inf)
        args.extend([G, h])
        if A is not None:
            A = cvxopt.matrix(A)
            b = cvxopt.matrix(b)
            args.extend([A, b])
        if initial_value is not None:
            sol = cvxopt.solvers.lp(*args, solver=solver, primal_start=initial_value)
        else:
            sol = cvxopt.solvers.lp(*args, solver=solver)

        if 'optimal' not in sol['status']:
            raise ValueError("LP optimum not found: %s" % sol['status'])

        if not return_only_solution:
            dual_variables = {}
            for idx, constr in enumerate(self.constraints):
                if self.constraints_map[idx]['type'] == 'affine_equality':
                    dual_variables[idx] = np.array(sol['y'])[self.constraints_map[idx]['indices']]
                elif self.constraints_map[idx]['type'] == 'affine_inequality':
                    dual_variables[idx] = np.array(sol['z'])[self.constraints_map[idx]['indices']]

            if sol['status'] == 'optimal':
                status = 'solved'
            else:
                status = 'unknown'

            output_dict = {
                'solution': np.array(sol['x']).reshape(self.input_shape),
                'status': status,
                'dual_variables': dual_variables
            }
            return output_dict
        else:
            return np.array(sol['x']).reshape(self.input_shape)

    def solve(self, initial_value: np.ndarray = None, solver='glpk', return_only_solution: bool = True):
        """Solve the problem

        Args:
            initial_value (numpy.ndarray), optional): Initial value for warm start. Defaults to None.
            solver (str, optional): Solver to use ['glpk', 'cvxopt']. Defaults to 'glpk'.

        Raises:
            ValueError: Unsupported solver

        Returns:
            numpy.ndarray: solution
        """
        if solver in ['glpk', 'cvxopt']:
            return self.__solve_cvxopt(initial_value=initial_value, solver=solver, return_only_solution=return_only_solution)
        else:
            raise ValueError("Unsupported solver")
