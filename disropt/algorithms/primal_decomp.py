import numpy as np
from typing import Union, Callable
from ..agents.agent import Agent
from ..problems.problem import Problem
from ..problems.constraint_coupled_problem import ConstraintCoupledProblem
from .algorithm import Algorithm
from ..functions import Variable
from ..functions import ExtendedFunction
from ..constraints import ExtendedConstraint


class PrimalDecomposition(Algorithm):
    """Distributed primal decomposition.

    From the perspective of agent :math:`i` the algorithm works as follows.

    Initialization: :math:`y_i^0` such that :math:`\sum_{i=1}^N y_i^0 = 0`
    
    For :math:`k=0,1,\\dots`

    * Compute :math:`((x_i^k, \\rho_i^k), \mu_i^k)` as a primal-dual optimal solution of

    .. math::
        :nowrap:

        \\begin{split}
        \min_{x_i, \\rho_i} \hspace{1.1cm} & \: f_i(x_i) + M \\rho_i \\\\
        \mathrm{subj. to} \: \mu_i: \: & \: g_i(x_i) \le y_i^k + \\rho_i \\boldsymbol{1} \\\\
        & \: x_i \in X_i, \\rho_i \ge 0
        \\end{split}
    
    * Gather :math:`\mu_j^k` from :math:`j \in \mathcal{N}_i` and update
    
    .. math::
        y_i^{k+1} = y_i^{k} + \\alpha^k \sum_{j \in \mathcal{N}_i} (\mu_i^k - \mu_j^k)

    where :math:`x_i\\in\\mathbb{R}^{n_i}`, :math:`\mu_i,y_i\\in\\mathbb{R}^S`, :math:`X_i\\subseteq\mathbb{R}^{n_i}` for all :math:`i` and :math:`\\alpha^k` is a positive stepsize.

    The algorithm has been presented in ????.
    """

    # TODO choose ref
    def __init__(self, agent: Agent, initial_condition: np.ndarray, enable_log: bool = False):
        super(PrimalDecomposition, self).__init__(agent, enable_log)

        if not isinstance(agent.problem, ConstraintCoupledProblem):
            raise TypeError("The agent must be equipped with a ConstraintCoupledProblem")
        
        if sum(1 for i in agent.problem.objective_function.input_shape if i > 1) > 1:
            raise ValueError("Currently only mono-dimensional objective functions are supported")

        if sum(1 for i in initial_condition.shape if i > 1) > 1:
            raise ValueError("Currently only mono-dimensional outputs for coupling functions are supported")

        # shape of local variable and coupling constraints
        self.x_shape = agent.problem.objective_function.input_shape
        self.S = initial_condition.size # TODO extend to non mono-dimensional case

        # initialize allocation and primal solution
        self.y0 = initial_condition
        self.y = initial_condition
        self.x = None

        # extended versions of objective function, coupling function and constraints (+ 1 variable for rho)
        self.objective_function = ExtendedFunction(agent.problem.objective_function)
        self.coupling_function = ExtendedFunction(agent.problem.coupling_function)
        self.local_constraints = ExtendedConstraint(agent.problem.constraints)

    def run(self, iterations: int = 1000, stepsize: Union[float, Callable] = 0.1, M: float = 1000.0, verbose: bool=False, **kwargs) -> np.ndarray:
        """Run the algorithm for a given number of iterations

        Args:
            iterations: Number of iterations. Defaults to 1000.
            stepsize: If a float is given as input, the stepsize is constant. 
                                                         If a function is given, it must take an iteration k as input and output the corresponding stepsize.. Defaults to 0.1.
            M: Value of the parameter :math:`M`. Defaults to 1000.
            verbose: If True print some information during the evolution of the algorithm. Defaults to False.

        Raises:
            TypeError: The number of iterations must be an int
            TypeError: The stepsize must be a float or a callable
            TypeError: The parameter M must be a float

        Returns:
            return a tuple (x, y) with the sequence of primal solutions and allocation estimates if enable_log=True.
        """
        if not isinstance(iterations, int):
            raise TypeError("The number of iterations must be an int")
        if not (isinstance(stepsize, float) or callable(stepsize)):
            raise TypeError("The stepsize must be a float or a function")
        if not isinstance(M, float):
            raise TypeError("The parameter M must be a float")

        if self.enable_log:
            # initialize sequence of x
            x_dims = [iterations]
            for dim in self.x_shape:
                x_dims.append(dim)
            self.x_sequence = np.zeros(x_dims)

            # initialize sequence of y
            y_dims = [iterations]
            for dim in self.y.shape:
                y_dims.append(dim)
            self.y_sequence = np.zeros(y_dims)

        for k in range(iterations):
            # store current allocation
            if self.enable_log:
                self.y_sequence[k] = self.y

            # perform an iteration
            if not isinstance(stepsize, float):
                step = stepsize(k)
            else:
                step = stepsize

            self.iterate_run(stepsize=step, M=M, **kwargs)

            # store primal solution
            if self.enable_log:
                self.x_sequence[k] = self.x
            
            if verbose:
                if self.agent.id == 0:
                    print('Iteration {}'.format(k), end="\r")

        if self.enable_log:
            return (self.x_sequence, self.y_sequence)
    
    def _update_local_solution(self, x: np.ndarray, mu: np.ndarray, mu_neigh: list, stepsize: float, **kwargs):
        """Update the local solution
        
        Args:
            x: current primal solution
            mu: current dual solution
            mu_neigh: dual solutions of neighbors
            stepsize: step-size for update
        """

        y_new = self.y

        for mu_j in mu_neigh:
            y_new += stepsize * (mu - mu_j)
        
        self.x = x
        self.y = y_new
    
    def iterate_run(self, stepsize: float, M: float, **kwargs):
        """Run a single iterate of the algorithm
        """
        # TODO extend this function to non mono-dimensional case

        # build local problem
        z = Variable(self.x_shape[0] + 1)
        A = np.hstack((np.zeros((self.S, self.x_shape[0])), np.ones((self.S,1)))).transpose()
        A_rho = np.hstack((np.zeros((1, self.x_shape[0])), [[1]])).transpose()
        rho = A_rho @ z
        alloc_constr = self.coupling_function <= self.y + A @ z
        rho_constr = rho >= 0
        constraints = [alloc_constr, rho_constr]
        constraints.extend(self.local_constraints)
        obj_function = self.objective_function + M * rho
        pb = Problem(obj_function, constraints)

        # solve problem and save data
        out = pb.solve(return_only_solution=False)
        if out['status'] != 'solved':
            raise ValueError("The local problem could not be solved")

        x = out['solution'][:-1]
        # rho = out['solution'][-1]
        mu = out['dual_variables'][0]

        # exchange dual variables with neighbors
        data = self.agent.neighbors_exchange(mu)
        mu_neigh = [data[idx] for idx in data ]

        self._update_local_solution(x, mu, mu_neigh, stepsize, **kwargs)
    
    def get_result(self):
        """Return the current value primal solution and allocation
    
        Returns:
            tuple (primal, allocation) of numpy.ndarray: value of primal solution, value of allocation
        """
        return (self.x, self.y)