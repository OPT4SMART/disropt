import numpy as np
from .linear_problem import LinearProblem
from ..functions import AffineForm
from ..functions.affine_form import aggregate_affine_form
from ..constraints import Constraint, AbstractSet
from cvxopt.glpk import ilp
from cvxopt import matrix


class MixedIntegerLinearProblem(LinearProblem):
    """Solve a Mixed-Integer Linear programming problem defined as:

    .. math::

        \\text{minimize } & c^\\top x

        \\text{subject to } & G x \\leq h

                            & A x = b

                            & x_k \\in \\mathbb{Z}, \\forall k \\in I

                            & x_k \\in \\{0,1\\}, \\forall k \\in B
        
    where :math:`I` is the set of integer variable indexes 
    and :math:`B` is the set of binary variable indexes.
    """

    def __init__(self, objective_function: AffineForm, constraints: list = None, integer_vars: list = None, binary_vars: list = None, **kwargs):
        """[summary]

        Args:
            objective_function (AffineForm): [description]
            constraints (list, optional): [description]. Defaults to None.
            integer_vars (list, optional)
            binary_vars (list, optional)
        """
        super().__init__(objective_function=objective_function, constraints=constraints, **kwargs)
        self.integer_vars = []
        self.binary_vars = []
        if integer_vars is not None:
            if not isinstance(integer_vars, list):
                integer_vars = [integer_vars]
            if any(isinstance(el, list) for el in integer_vars):
                raise TypeError("integer_vars list must not contain sub-lists")
            for int_idx in integer_vars:
                if not isinstance(int_idx, int) or int_idx < 0:
                    raise TypeError("integer_vars must be a list of non-negative indexes")
                self.integer_vars.append(int_idx)
        if binary_vars is not None:
            if not isinstance(binary_vars, list):
                binary_vars = [binary_vars]
            if any(isinstance(el, list) for el in binary_vars):
                raise TypeError("binary_vars must not contain sub-lists")
            for bin_idx in binary_vars:
                if not isinstance(bin_idx, int) or bin_idx < 0:
                    raise TypeError("binary_vars must be a list of non-negative indexes")
                self.binary_vars.append(bin_idx)

    def __get_cvxopt_objective_parameters(self):
        """Return the objective parameter required by cvxopt

        Returns:
            numpy.ndarray
        """
        c,  _ = self.objective_function.get_parameters()
        return c

    def __get_gurobi_objective_parameters(self):
        """Return the objective parameter required by cvxopt

        Returns:
            numpy.ndarray
        """
        c,  _ = self.objective_function.get_parameters()
        return c.flatten()

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
                    A = matrix(A, tc='d')
                    b = matrix(b, tc='d')

                if const.is_inequality:
                    G, h = const.get_parameters()
                    G = G.transpose()
                    h = -h
                    G = matrix(G, tc='d')
                    h = matrix(h, tc='d')

        return G, h, A, b

    def __get_gurobi_constraints_parameters(self):
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

    def __solve_cvxopt(self, solver='glpk', return_only_solution: bool = True):
        """Solve the problem through cvxopt interface

        Args:
            return_only_solution (bool): if True, returns only the optimal variable. If False, returns also the optimization status

        Raises:
            ValueError: solver must be 'glpk'
            ValueError: optimal solution not found

        Returns:
            numpy.ndarray: solution
        """
        try:
            import cvxopt
        except ImportError:
            raise ImportError("CVXOPT is required")
        try:
            import cvxopt.glpk
            options = dict(msg_lev='GLP_MSG_OFF')
        except ImportError:
            print("GLPK solver not found.")

        c = self.__get_cvxopt_objective_parameters()
        G, h, A, b = self.__get_cvxopt_constraints_parameters()

        sol = ilp(c, G, h, A, b, set(self.integer_vars), set(self.binary_vars), options)

        if sol[0] != 'optimal':
            raise ValueError("MILP optimum not found: %s" % sol[0])

        if not return_only_solution:
            if sol[0] == 'optimal':
                status = 'solved'
            else:
                status = 'unknown'

            output_dict = {
                'solution': np.array(sol[1]).reshape(self.input_shape),
                'status': status,
            }
            return output_dict
        else:
            return np.array(sol[1]).reshape(self.input_shape)

    def __solve_gurobi(self, initial_value: np.ndarray = None, return_only_solution: bool = True):
        """Solve the problem through gurobi

        Args:
            initial_value (np.ndarray, optional): starting point for warm start (if possible). Defaults to None.
            return_only_solution (bool): if True, returns only the optimal variable. If False, returns also the optimization status

        Raises:
            ImportError: Gurobi is required
            ValueError: optimal solution not found

        Returns:
            numpy.ndarray: solution
        """
        try:
            import gurobipy as gb
            from gurobipy import GRB
        except ImportError:
            raise ImportError("Gurobi is required")

        # create the model and variables
        model = gb.Model("milp")
        x = model.addMVar(self.input_shape[0], lb=-GRB.INFINITY)
        # set integer and binary constraints
        if self.integer_vars is not None:
            for ii in self.integer_vars:
                x[ii].setAttr(GRB.Attr.VType, GRB.INTEGER)
        if self.binary_vars is not None:
            for ii in self.binary_vars:
                x[ii].setAttr(GRB.Attr.VType, GRB.BINARY)
        if initial_value is not None:
            if not isinstance(initial_value, np.ndarray):
                raise TypeError("initial_value must be an instance of numpy.ndarray")
            if initial_value.ndim > 2 or (initial_value.ndim == 2 and initial_value.shape[1] > 1):
                raise ValueError("initial_value must be 1-dimensional or a 2-dimensional column vector")
            if initial_value.shape[0] != self.input_shape[0]:
                raise ValueError("The number of entries of initial_value and of the optimization variable must coincide")
            x[ii].setAttr("Start", initial_value[ii])

        # set objective function
        c = self.__get_gurobi_objective_parameters()
        model.setMObjective(None, c, 0.0, None, None, x, GRB.MINIMIZE)
        # set constraints and update the model
        G, h, A, b = self.__get_gurobi_constraints_parameters()
        if A is not None:
            model.addMConstrs(A, x, '=', b)
        if G is not None:
            model.addMConstrs(G, x, '<', h)

        model.setParam('OutputFlag', 0)
        model.update()

        model.optimize()

        sol = model.getAttr(GRB.Attr.Status)
        x_star = model.getAttr(GRB.Attr.X)
        if sol != 2:
            raise ValueError("MILP optimum not found: %d" % sol)
        if not return_only_solution:
            if sol == 2:
                status = 'solved'
            else:
                status = 'unknown'

            output_dict = {
                'solution': np.array(x_star).reshape(self.input_shape),
                'status': status,
            }
            return output_dict
        else:
            return np.array(x_star).reshape(self.input_shape)

    def solve(self, initial_value: np.ndarray = None, solver='glpk', return_only_solution: bool = True):
        """Solve the problem

        Args:
            initial_value (numpy.ndarray), optional): Initial value for warm start. Defaults to None. Not available in GLPK
            solver (str, optional): Solver to use ['glpk', 'gurobi']. Defaults to 'glpk'.

        Raises:
            ValueError: Unsupported solver

        Returns:
            numpy.ndarray: solution
        """

        if solver in ['glpk']:
            return self.__solve_cvxopt(return_only_solution=return_only_solution)
        elif solver in ['gurobi']:
            return self.__solve_gurobi(initial_value=initial_value, return_only_solution=return_only_solution)
        else:
            raise ValueError("Unsupported solver")
