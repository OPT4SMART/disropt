import numpy as np
from . import MixedIntegerLinearProblem, LinearProblem
from ..functions import AffineForm, Variable
from ..functions.affine_form import aggregate_affine_form
from ..constraints import Constraint, AbstractSet


class ConvexifiedMILP(MixedIntegerLinearProblem):
    """Solve a convexified Mixed-Integer Linear Problem of the form:

    .. math::

        \\text{minimize } & c^\\top x + M \\rho

        \\text{subject to } & x \\in \mathrm{conv}(X), \:\\rho \\geq 0

                            & Ax \\leq y + \\rho \mathbf{1}
        
    where the set :math:`X` is a compact mixed-integer polyhedral set defined by equality
    and inequality constraints. Moreover, :math:`\\rho` is a scalar,
    :math:`y \in \mathbb{R}^{m}` is a vector and :math:`A \in \mathbb{R}^{m \\times n}`
    is a constraint matrix.
    """

    def __init__(self, objective_function: AffineForm, y: np.ndarray, A: np.ndarray,
        constraints: list = None, integer_vars: list = None, binary_vars: list = None, **kwargs):
        """initialize the class

        Args:
            objective_function (AffineForm): linear cost function :math:`c^\\top x`
            y (np.ndarray): value of the vector :math:`y`
            A (np.ndarray): constraint matrix
            constraints (list): equality and inequality constraints describing the set :math:`X`
            integer_vars (list, optional): list of integer variables in the set :math:`X`
            binary_vars (list, optional): list of binary variables in the set :math:`X`
        """

        super().__init__(objective_function=objective_function, constraints=constraints,
            integer_vars=integer_vars, binary_vars=binary_vars, **kwargs)

        # check inputs
        if not isinstance(y, np.ndarray):
            raise TypeError("y must be an instance of numpy.ndarray")
        if not isinstance(A, np.ndarray):
            raise TypeError("A must be an instance of numpy.ndarray")
        
        # check dimensions
        if y.ndim > 2 or (y.ndim == 2 and y.shape[1] > 1):
            raise ValueError("y must be 1-dimensional or a 2-dimensional column vector")
        if A.ndim != 2:
            raise ValueError("A must be 2-dimensional matrix")
        if y.shape[0] != A.shape[0]:
            raise ValueError("The number of rows of y and A must coincide")
        if objective_function.input_shape[0] != A.shape[1]:
            raise ValueError("The number of variables in objective_function must coincide with the number of columns of A")

        # save data
        self.nvars   = A.shape[1] # number of optimization variables
        self.nconstr = A.shape[0] # number of inequality constraints
        self.y = np.copy(y)
        self.A = np.copy(A)
        if y.ndim == 1:
            self.y = self.y[:, None]

        # initialize more data
        self.last_mu = np.zeros((self.nconstr, 1))
        self.x = Variable(self.nvars) # symbolic primal variable
        self.dual_var = Variable(self.nconstr + 1) # symbolic dual variable: [mu, epigraph]

        # prepare MILP problem
        self.c = objective_function.get_parameters()[0] # primal cost vector
        self.lagr_cost = (self.c + self.A.T @ self.last_mu) @ self.x # cost function of Lagrangian
        self.milp = MixedIntegerLinearProblem(objective_function=self.lagr_cost, constraints=constraints,
            integer_vars=integer_vars, binary_vars=binary_vars) # lagrangian MILP problem
        
        # prepare LP problem
        self.lp_cost = np.vstack((self.y, 1)) @ self.dual_var
    
    def solve(self, M: float, milp_solver=None, initial_dual_solution: np.ndarray = None,
        return_only_solution: bool = True, y: np.ndarray = None, max_cutting_planes: int = None,
        cutting_planes: np.ndarray = None, threshold_convergence: float = 1e-8, max_iterations: int = 1000):
        """Solve the problem using a custom dual cutting-plane algorithm

        Args:
            M (float): value of the parameter :math:`M`
            milp_solver (str, optional): MILP solver to use. Defaults to None (use default solver).
            initial_dual_solution (numpy.ndarray, optional): Initial dual value for warm start. Defaults to None.
            return_only_solution (bool, optional): if True, returns only solution, otherwise returns more information. Defaults to True.
            y (np.ndarray, optional): value to override the current y. Defaults to None (keep current value).
            max_cutting_planes (int, optional): maximum number of stored cutting planes. Defaults to None (disabled).
            cutting_planes (numpy.ndarray, optional): cutting planes for warm start previously returned by this function. Defaults to None.
            threshold_convergence (float, optional): threshold for convergence detection. Defaults to 1e-8.
            max_iterations (int, optional): maximum number of iterations performed by algorithm. Defaults to 1e3.

        Returns:
            tuple: primal solution tuple (x, rho) if return_only_solution = True,
            otherwise (primal solution tuple, optimal cost, dual solution, cutting planes, number of iterations)
        """

        # check parameters
        if max_cutting_planes is not None and (not isinstance(max_cutting_planes, int) or max_cutting_planes <= 0):
            raise TypeError("max_cutting_planes must be either None or a positive integer")
        if not isinstance(max_iterations, int) or max_iterations <= 0:
            raise TypeError("max_iterations must be a positive integer")
        if not isinstance(threshold_convergence, float) or threshold_convergence <= 0:
            raise TypeError("threshold_convergence must be a positive float")
        if milp_solver is not None and not isinstance(milp_solver, str):
            raise TypeError("milp_solver must be either None or a string")
        if not isinstance(M, float) or M < 0:
            raise TypeError("M must be a non-negative float")

        # check initial dual solution
        if initial_dual_solution is not None:
            if not isinstance(initial_dual_solution, np.ndarray):
                raise TypeError("initial_dual_solution must be either None or a numpy.ndarray")
            if initial_dual_solution.ndim > 2 or (initial_dual_solution.ndim == 2 and initial_dual_solution.shape[1] > 1):
                raise ValueError("initial_dual_solution must be 1-dimensional or a 2-dimensional column vector")
            if initial_dual_solution.shape[0] != self.nconstr:
                raise ValueError("The number of rows of initial_dual_solution and A must coincide")

            if initial_dual_solution.ndim == 1:
                initial_dual_solution = np.copy(initial_dual_solution)[:, None]
            else:
                initial_dual_solution = np.copy(initial_dual_solution)

        # check y
        if y is not None:
            if not isinstance(y, np.ndarray):
                raise TypeError("y must be an instance of numpy.ndarray")
            if y.ndim > 2 or (y.ndim == 2 and y.shape[1] > 1):
                raise ValueError("y must be 1-dimensional or a 2-dimensional column vector")
            if y.shape[0] != self.nconstr:
                raise ValueError("The number of rows of y and A must coincide")

            if y.ndim == 1:
                self.y = np.copy(y)[:, None]
            else:
                self.y = np.copy(y)
        
        # check cutting planes
        # cutting planes are row vectors of the form [A, b, x^T]
        # (each one represents a constraint Ax <= b obtained at MILP solution x)
        n_cut = 0
        if cutting_planes is not None:
            if not isinstance(cutting_planes, np.ndarray):
                raise TypeError("cutting_planes must be either None or a numpy.ndarray")
            if cutting_planes.ndim != 2:
                raise ValueError("cutting_planes must be a 2-dimensional matrix")
            if cutting_planes.shape[1] != self.nconstr + self.nvars + 2:
                raise ValueError("The number of columns of cutting_planes must coincide with the sum of sizes of A + 2")

            n_cut = cutting_planes.shape[0]

        # initialize dual variable
        if initial_dual_solution is not None:
            # project onto positive orthant
            mu = np.maximum(initial_dual_solution, np.zeros((self.nconstr, 1)))
        else:
            mu = self.last_mu
        
        # check maximum number of allowed cutting planes
        n_allowed_cuts = max_iterations 
        if max_cutting_planes is not None:
            n_allowed_cuts = min(max_iterations, max_cutting_planes)

            # discard excess cutting planes if necessary
            n_cut = min(n_cut, n_allowed_cuts-1) # pro tip: -1 to leave one empty slot for initialization

        # initialize storage variables
        x_store = np.zeros((self.nvars, n_allowed_cuts))
        A_cuts  = np.zeros((n_allowed_cuts, self.nconstr + 1)) # cutting planes
        b_cuts  = np.zeros((n_allowed_cuts, 1))

        # constraints mu^T 1 <= M and mu >= 0
        self.M = M
        basic_lp_constraints = [
                np.column_stack((np.ones((1, self.nconstr)), 0)).T @ self.dual_var <= self.M,
                np.column_stack((-np.eye(self.nconstr), np.zeros((self.nconstr, 1)))).T @ self.dual_var <= 0
        ]
        
        # initialize cutting planes
        if cutting_planes is not None:
            # take only the last n_cut cuts
            A_cuts[0:n_cut, :]  = cutting_planes[-n_cut:,0:self.nconstr+1]
            b_cuts[0:n_cut]     = cutting_planes[-n_cut:,self.nconstr+1:self.nconstr+2]
            x_store[:, 0:n_cut] = cutting_planes[-n_cut:,-self.nvars:].T
        
        # initialize LP problem
        self.lp_cost = np.row_stack((self.y, 1)) @ self.dual_var
        
        ###########################################################

        # solve initial MILP
        self.lagr_cost = (self.c + self.A.T @ mu) @ self.x
        self.milp.objective_function = self.lagr_cost
        x = self.milp.solve() if milp_solver is None else self.milp.solve(solver=milp_solver)
        
        x_store[:, n_cut] = x.flatten()
        A_cuts[n_cut, :]  = np.column_stack((-(self.A @ x).T, -1))
        b_cuts[n_cut]     = self.c.T @ x
        true_cost         = self.lagr_cost.eval(x) - mu.T@self.y

        # perform iterations
        for k in range(max_iterations):
            # compute last index of cutting planes that can be used
            index_k = min(n_cut+k+1, n_allowed_cuts)

            # solve LP
            lp_constr = [A_cuts[0:index_k, :].T @ self.dual_var <= b_cuts[0:index_k]]
            lp_constr.extend(basic_lp_constraints)
            lp = LinearProblem(objective_function=self.lp_cost, constraints=lp_constr)
            solution = lp.solve(return_only_solution=False)
            mu = solution['solution'][:-1] # mu variable
            multipliers = solution['dual_variables']
            approx_cost = -self.lp_cost.eval(solution['solution'])

            # solve MILP
            self.lagr_cost = (self.c + self.A.T @ mu) @ self.x
            self.milp.objective_function = self.lagr_cost
            x = self.milp.solve() if milp_solver is None else self.milp.solve(solver=milp_solver)
            
            true_cost = self.lagr_cost.eval(x) - mu.T@self.y
            
            # check stopping criterion
            if (abs(true_cost - approx_cost) < threshold_convergence):
                break
            
            # in case of saturation, free a slot for next cutting plane
            if index_k >= n_allowed_cuts:
                A_cuts  = np.roll(A_cuts, -1, axis=0)
                b_cuts  = np.roll(b_cuts, -1, axis=0)
                x_store = np.roll(x_store, -1, axis=1)
                index_k -= 1
            
            # store new cutting plane
            x_store[:, index_k] = x.flatten()
            A_cuts[index_k, :]  = np.column_stack((-(self.A @ x).T, -1))
            b_cuts[index_k]     = self.c.T @ x
        
        # save data and compute primal solution
        self.last_mu = mu
        lambdas  = multipliers[0].flatten() # multipliers associated with cutting planes
        rho_star = np.asscalar(multipliers[1].flatten()) # multiplier associated with mu^T 1 <= M
        x_star   = sum([lambdas[i]*x_store[:, i] for i in range(index_k)]) # primal solution
        x_star   = x_star[:, None]

        # prepare cutting planes to return (duplicates are removed)
        cutting_planes = np.column_stack((A_cuts[0:index_k, :], b_cuts[0:index_k, :], x_store[:, 0:index_k].T))
        unique_ind     = np.unique(cutting_planes, axis=0, return_index=True)[1]
        cutting_planes = cutting_planes[unique_ind, :]

        # return solution
        if not return_only_solution:
            # TODO unify this return with the returns of other files
            # return a tuple with (primal solution, optimal cost, dual solution, cutting planes, number of iterations)
            return ((x_star, rho_star), np.asscalar(true_cost), mu, cutting_planes, k+1)
        else:
            # return only primal solution (x, rho)
            return (x_star, rho_star)