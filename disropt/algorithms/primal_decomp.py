import numpy as np
from typing import Union, Callable
from threading import Event
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

        # initialize allocation, primal solution and cost
        self.y0 = np.copy(initial_condition)
        self.y = np.copy(initial_condition)
        self.y_avg = np.copy(initial_condition)
        self.x = None
        self.J = None

        # extended versions of objective function, coupling function and constraints (+ 1 variable for rho)
        self.objective_function = ExtendedFunction(agent.problem.objective_function)
        self.coupling_function = ExtendedFunction(agent.problem.coupling_function)
        self.local_constraints = ExtendedConstraint(agent.problem.constraints)

    def run(self, iterations: int = 1000, stepsize: Union[float, Callable] = 0.1, M: float = 1000.0,
        verbose: bool=False, callback_iter: Callable=None, compute_runavg: bool=False, runavg_start_iter: int=0,
        event: Event=None, **kwargs) -> np.ndarray:
        """Run the algorithm for a given number of iterations

        Args:
            iterations: Number of iterations. Defaults to 1000.
            stepsize: If a float is given as input, the stepsize is constant. 
                                                         If a function is given, it must take an iteration k as input and output the corresponding stepsize.. Defaults to 0.1.
            M: Value of the parameter :math:`M`. Defaults to 1000.
            verbose: If True print some information during the evolution of the algorithm. Defaults to False.
            callback_iter: callback function to be called at the end of each iteration. Must take an iteration k as input. Defaults to None.
            compute_runavg: whether or not to compute also running average of allocation. Defaults to False.
            runavg_start_iter: specifies when to start computing running average (applies only if compute_runavg = True). Defaults to 0.

        Raises:
            TypeError: The number of iterations must be an int
            TypeError: The stepsize must be a float or a callable
            TypeError: The parameter M must be a float

        Returns:
            return a tuple (x, y, J) with the sequence of primal solutionsm allocation estimates and cost if enable_log=True. If compute_runavg=True, then return (x, y, y_avg, J)
        """
        if not isinstance(iterations, int):
            raise TypeError("The number of iterations must be an int")
        if not (isinstance(stepsize, float) or callable(stepsize)):
            raise TypeError("The stepsize must be a float or a function")
        if not isinstance(M, float):
            raise TypeError("The parameter M must be a float")
        if callback_iter is not None and not callable(callback_iter):
            raise TypeError("The callback function must be a Callable")
        if runavg_start_iter < 0:
            raise ValueError("The parameter runavg_start_iter must not be negative")

        if self.enable_log:
            # initialize sequence of costs
            x_dims = [iterations]
            self.J_sequence = np.zeros(x_dims)

            # initialize sequence of x
            for dim in self.x_shape:
                x_dims.append(dim)
            self.x_sequence = np.zeros(x_dims)

            # initialize sequence of y
            y_dims = [iterations]
            for dim in self.y.shape:
                y_dims.append(dim)
            self.y_sequence = np.zeros(y_dims)

            # initialize sequence of averaged y
            if compute_runavg:
                self.y_avg_sequence = np.zeros(y_dims)

        # initialize cumulative sum of stepsize if needed
        if compute_runavg:
            self.stepsize_sum = 0
        
        last_iter = np.copy(iterations)

        for k in range(iterations):
            # store current allocation
            if self.enable_log:
                self.y_sequence[k] = self.y

                if compute_runavg:
                    self.y_avg_sequence[k] = self.y_avg

            # compute stepsize
            if not isinstance(stepsize, float):
                step = stepsize(k)
            else:
                step = stepsize

            # determine whether or not running average must be updated
            update_runavg = compute_runavg and k >= runavg_start_iter

            # perform iteration
            self.iterate_run(stepsize=step, M=M, update_runavg=update_runavg, event=event, **kwargs)

            # store primal solution and cost
            if self.enable_log:
                self.x_sequence[k] = self.x
                self.J_sequence[k] = self.J
            
            # check if we must stop for external event
            if event is not None and event.is_set():
                last_iter = k
                break
            
            # perform external callback (if any)
            if callback_iter is not None:
                callback_iter(k)
            
            if verbose:
                if self.agent.id == 0:
                    print('Iteration {}'.format(k), end="\r")

        if self.enable_log and compute_runavg:
            return (self.x_sequence.take(np.arange(0,last_iter), axis=0), self.y_sequence.take(np.arange(0,last_iter), axis=0),
                self.y_avg_sequence.take(np.arange(0,last_iter), axis=0), self.J_sequence[0:last_iter])
        elif self.enable_log:
            return (self.x_sequence.take(np.arange(0,last_iter), axis=0), self.y_sequence.take(np.arange(0,last_iter), axis=0),
                self.J_sequence[0:last_iter])
    
    def _update_local_solution(self, x: np.ndarray, mu: np.ndarray, mu_neigh: list, stepsize: float, update_runavg, **kwargs):
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
        self.J = np.asscalar(self.agent.problem.objective_function.eval(x))
        self.y = y_new

        if update_runavg:
            self.stepsize_sum += stepsize
            self.y_avg += stepsize * (self.y - self.y_avg) / self.stepsize_sum
    
    def iterate_run(self, stepsize: float, M: float, update_runavg, event: Event, **kwargs):
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
        data = self.agent.neighbors_exchange(mu, event=event)
        mu_neigh = [data[idx] for idx in data]

        if event is None or not event.is_set():
            self._update_local_solution(x, mu, mu_neigh, stepsize, update_runavg, **kwargs)
    
    def get_result(self, return_runavg:bool = False):
        """Return the current value of primal solution, allocation and cost

        Args:
            return_runavg: whether or not to return also running average of allocation. Defaults to False.
    
        Returns:
            tuple (primal, allocation, cost) of numpy.ndarray: value of primal solution, allocation, cost (if return_runavg = False)
            tuple (primal, allocation, allocation_avg cost) if return_runavg = True
        """
        if return_runavg:
            return (self.x, self.y, self.y_avg, self.J)
        else:
            return (self.x, self.y, self.J)