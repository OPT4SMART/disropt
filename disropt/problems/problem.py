import numpy as np
import warnings
from typing import Union
from ..functions import AbstractFunction
from ..functions.affine_form import aggregate_affine_form
from ..constraints import Constraint, AbstractSet
from .utilities import check_affine_constraints
from ..utils.utilities import check_symmetric, is_semi_pos_def


class Problem:
    """A generic optimization problem. 

    .. math::

        \\text{minimize } & f(x)

        \\text{subject to } & g(x) \\leq 0

                            & h(x) = 0

    Args:
        objective_function (AbstractFunction, optional): objective function. Defaults to None.
        constraints (list, optional): constraints. Defaults to None.

    Attributes:
        objective_function (Function): Objective function to be minimized
        constraints (list): constraints
        input_shape (tuple): dimension of optimization variable
        output_shape (tuple): output shape
    """

    def __new__(cls, objective_function: AbstractFunction = None, constraints: list = None, force_general_problem: bool = False):
        instance = object.__new__(cls)
        if not force_general_problem:
            if cls.__name__ == 'Problem':
                if objective_function is not None:  # TODO feasibility problem
                    if objective_function.is_affine:
                        if check_affine_constraints(constraints):
                            # Convert to LinearProblem
                            from .linear_problem import LinearProblem
                            instance = LinearProblem(objective_function, constraints)

                    elif objective_function.is_quadratic:
                        P, _, _ = objective_function.get_parameters()
                        if is_semi_pos_def(P):
                            if check_affine_constraints(constraints):
                                # Convert to QuadraticProblem
                                from .quadratic_problem import QuadraticProblem
                                instance = QuadraticProblem(objective_function, constraints, is_pos_def=True)
        return instance

    def __init__(self, objective_function: AbstractFunction = None, constraints: list = None,
                 force_general_problem: bool = False):
        self.objective_function = None
        self.constraints = []
        self.input_shape = None
        self.output_shape = None
        if objective_function is not None:
            self.set_objective_function(objective_function)

        if constraints is not None:
            if not isinstance(constraints, list):
                constraints = [constraints]
            for constraint in constraints:
                self.add_constraint(constraint)

        self.__split_constraints()

    def set_objective_function(self, fn: AbstractFunction):
        """Set the objective function

        Args:
            fn: objective function

        Raises:
            TypeError: input must be a AbstractFunction object
        """
        if isinstance(fn, AbstractFunction):
            self.objective_function = fn
            self.input_shape = fn.input_shape
            self.output_shape = fn.output_shape
        else:
            raise TypeError("input must be a AbstractFunction")

    def add_constraint(self, fn: Union[AbstractSet, Constraint]):
        """Add a new constraint

        Args:
            fn: new constraint

        Raises:
            TypeError: constraints must be AbstractSet or Constraint
        """
        if isinstance(fn, Constraint):
            self.constraints.append(fn)
        elif isinstance(fn, AbstractSet):
            constraints_list = fn.to_constraints()
            for const in constraints_list:
                self.constraints.append(const)
        else:
            raise TypeError("constraints must be AbstractSet or Constraint")

    def __split_constraints(self):
        """split the constraints into affine equalities, affine inequalities and nonlinear constraints

        Raises:
            ValueError: Nonlinear constraints must have scalar output
        """
        affine_equalities = []
        affine_inequalities = []
        self.affine_equality = None
        self.affine_inequality = None
        self.nonlinear_constraints = []
        self.constraints_map = {}
        equality_idx_counter = 0
        inequality_idx_counter = 0
        nonlinear_idx_counter = 0
        for idx, constr in enumerate(self.constraints):
            if constr.is_affine:
                if constr.is_equality:
                    affine_equalities.append(constr.function)
                    self.constraints_map[idx] = {'type': 'affine_equality', 'indices': list(
                        range(equality_idx_counter, equality_idx_counter+constr.output_shape[0]))}
                    equality_idx_counter += constr.output_shape[0]
                else:
                    affine_inequalities.append(constr.function)
                    self.constraints_map[idx] = {'type': 'affine_inequality', 'indices': list(
                        range(inequality_idx_counter, inequality_idx_counter+constr.output_shape[0]))}
                    inequality_idx_counter += constr.output_shape[0]
            else:
                if constr.output_shape != (1, 1):
                    raise ValueError("Nonlinear constraints must have scalar output")
                if constr.is_inequality:
                    self.nonlinear_constraints.append(constr)
                    self.constraints_map[idx] = {'type': 'nonlinear_inequality', 'indices': list(
                        range(nonlinear_idx_counter, nonlinear_idx_counter+constr.output_shape[0]))}
                    nonlinear_idx_counter += constr.output_shape[0]
                else:
                    warnings.warn("Nonconvex problem")
                    c1 = constr.function <= 0
                    c2 = constr.function >= 0
                    self.nonlinear_constraints.append(c1)
                    self.nonlinear_constraints.append(c2)

        if len(affine_equalities) != 0:
            aggregated_equality = aggregate_affine_form(affine_equalities)
            self.affine_equality = aggregated_equality == 0

        if len(affine_inequalities) != 0:
            aggregated_inequality = aggregate_affine_form(affine_inequalities)
            self.affine_inequality = aggregated_inequality <= 0

    def __nonlinear_f_cvxopt(self, x=None, z=None):
        """nonlinear F function for cvxopt solver
        """
        try:
            import cvxopt
        except ImportError:
            ImportError("CVXOPT is required")

        m = len(self.nonlinear_constraints)
        if x is None and z is None:
            # TODO Check domain of functions
            x0 = np.random.randn(*self.input_shape)
            return m, cvxopt.matrix(x0)

        if z is None:
            # TODO check size of x=(n, 1)
            x_np = np.array(x)
            f = np.zeros([m + 1, 1])
            Df = np.zeros([m + 1, self.input_shape[0]])
            f[0] = self.objective_function.eval(x_np)
            Df[0, :] = self.objective_function.subgradient(x_np).flatten()
            for idx, constr in enumerate(self.nonlinear_constraints):
                f[idx + 1] = constr.function.eval(x_np)
                Df[idx + 1, :] = constr.function.subgradient(x_np).flatten()
            # TODO check domain and return None if x is not in the domain
            return cvxopt.matrix(f), cvxopt.matrix(Df)

        # TODO check size of x=(n, 1) and z=(m+1, 1)
        x_np = np.array(x)
        f = np.zeros([m + 1, 1])
        Df = np.zeros([m + 1, self.input_shape[0]])
        H = np.zeros([self.input_shape[0], self.input_shape[0]])
        f[0] = self.objective_function.eval(x_np)
        Df[0, :] = self.objective_function.subgradient(x_np).flatten()
        H += z[0] * self.objective_function.hessian(x_np)
        for idx, constr in enumerate(self.nonlinear_constraints):
            f[idx + 1] = constr.function.eval(x_np)
            Df[idx + 1, :] = constr.function.subgradient(x_np).flatten()
            H += z[idx+1] * constr.function.hessian(x_np)
        return cvxopt.matrix(f), cvxopt.matrix(Df), cvxopt.matrix(H)

    def solve(self, solver='cvxpy', return_only_solution: bool = True):
        """Solve the problem

        Returns:
            numpy.ndarray: solution
        """
        if solver == 'cvxpy':
            try:
                import cvxpy as cvx
            except ImportError:
                warnings.warn("CVXPY is not installed. Trying with CVXOPT.")
                solver = 'cvxopt'
            try:
                cvxpy_func = self.objective_function._to_cvxpy()
                obj = cvx.Minimize(cvxpy_func)
                constraints = []
                for const in self.constraints:
                    constraints.append(const._to_cvxpy())
                pb = cvx.Problem(obj, constraints)
                try:
                    pb.solve()
                except cvx.SolverError:
                    pb.solve(solver='SCS')
                except RuntimeError:
                    pb.solve(solver='ECOS')

                sol = np.array(pb.variables()[0].value).reshape(self.input_shape)
                if not return_only_solution:
                    if pb.status == 'optimal':
                        status = 'solved'
                    else:
                        status = pb.status
                    dual_variables = {}
                    for idx, const in enumerate(constraints):
                        dual_variables[idx] = np.array(const.dual_value).reshape(self.constraints[idx].output_shape)

                    output_dict = {
                        'solution': sol,
                        'status': status,
                        'dual_variables': dual_variables
                    }
                    return output_dict
                else:
                    return sol
            except RuntimeError:
                warnings.warn("CVXPY cannot solve the problem. Trying with CVXOPT.")
                solver = 'cvxopt'
            except:
                warnings.warn("CVXPY cannot solve the problem. Trying with CVXOPT.")
                solver = 'cvxopt'

        if solver == 'cvxopt':
            try:
                import cvxopt
            except ImportError:
                raise ImportError("CVXOPT is required")
            cvxopt.solvers.options['show_progress'] = False
            cvxopt.solvers.options['maxiters'] = 1000
            try:
                F = self.__nonlinear_f_cvxopt
                G = None
                h = None
                A = None
                b = None
                if self.affine_inequality is not None:
                    G, h = self.affine_inequality.get_parameters()
                    G = G.transpose()
                    G = cvxopt.matrix(G)
                    h = cvxopt.matrix(-h)
                if self.affine_equality is not None:
                    A, b = self.affine_equality.get_parameters()
                    A = A.transpose()
                    A = cvxopt.matrix(A)
                    b = cvxopt.matrix(-b)
                sol = cvxopt.solvers.cp(F, G=G, h=h, A=A, b=b)

                if not return_only_solution:
                    dual_variables = {}
                    for idx, _ in enumerate(self.constraints):
                        if self.constraints_map[idx]['type'] == 'affine_equality':
                            dual_variables[idx] = np.array(sol['y'])[self.constraints_map[idx]['indices']]
                        elif self.constraints_map[idx]['type'] == 'affine_inequality':
                            dual_variables[idx] = np.array(sol['zl'])[self.constraints_map[idx]['indices']]
                        elif self.constraints_map[idx]['type'] == 'nonlinear_inequality':
                            dual_variables[idx] = np.array(sol['znl'])[self.constraints_map[idx]['indices']]

                    if sol['status'] == 'optimal':
                        status = 'solved'
                    else:
                        status = sol['status']

                    output_dict = {
                        'solution': np.array(sol['x']).reshape(self.input_shape),
                        'status': status,
                        'dual_variables': dual_variables
                    }
                    return output_dict
                else:
                    return np.array(sol['x']).reshape(self.input_shape)
            except:
                raise ValueError("Cannot solve the problem")

    def project_on_constraint_set(self, x: np.ndarray) -> np.ndarray:
        """Compute the projection of a point onto the constraint set of the problem

        Args:
            x : point to project

        Returns:
            numpy.ndarray: projected point
        """
        from .projection_problem import ProjectionProblem
        pb = ProjectionProblem(self.constraints, x)
        return pb.solve()
