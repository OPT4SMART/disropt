import numpy as np
from typing import Union, Callable
from copy import deepcopy
from ..agents.agent import Agent
from ..problems.problem import Problem
from .algorithm import Algorithm
from ..functions import Variable


class DualDecomposition(Algorithm):
    """Distributed dual decomposition.

    From the perspective of agent :math:`i` the algorithm works as follows.

    Initialization: :math:`\lambda_{ij}^0` for all :math:`j \in \mathcal{N}_i`
    
    For :math:`k=0,1,\\dots`

    * Gather :math:`\lambda_{ji}^k` from neighbors :math:`j \in \mathcal{N}_i`

    * Compute :math:`x_i^k` as an optimal solution of

    .. math::
        \min_{x_i \in X_i} \: f_i(x_i) + x_i^\\top \sum_{j \in \mathcal{N}_i} (\lambda_{ij}^k - \lambda_{ji}^k)
    
    * Gather :math:`x_j^k` from neighbors :math:`j \in \mathcal{N}_i`
    
    * Update for all :math:`j \in \mathcal{N}_i`
    
    .. math::
        \lambda_{ij}^{k+1} = \lambda_{ij}^{k} + \\alpha^k (x_i^k - x_j^k)

    where :math:`x_i\\in\\mathbb{R}^{n}`, :math:`\lambda_{ij}\\in\\mathbb{R}^n` for all :math:`j \in \mathcal{N}_i`, :math:`X_i\\subseteq\mathbb{R}^{n}` for all :math:`i` and :math:`\\alpha^k` is a positive stepsize.

    The algorithm has been presented in ????.
    """

    # TODO choose ref
    def __init__(self, agent: Agent, initial_condition: dict, enable_log: bool = False):
        super(DualDecomposition, self).__init__(agent, enable_log)

        if not isinstance(agent.problem, Problem):
            raise TypeError("The agent must be equipped with a Problem")
        
        if sum(1 for i in agent.problem.objective_function.input_shape if i > 1) > 1:
            raise ValueError("Currently only mono-dimensional objective functions are supported")
        
        if not all([isinstance(x, np.ndarray) for _, x in initial_condition.items()]):
            raise TypeError("The initial condition dictionary can only contain numpy vectors")
        
        if sorted(agent.in_neighbors) != sorted([x for x in initial_condition]):
            raise TypeError("The initial condition dictionary must contain exactly one vector per neighbor")

        # shape of local variable
        self.x_shape = agent.problem.objective_function.input_shape

        # initialize dual variables and primal solution
        self.lambd0 = deepcopy(initial_condition)
        self.lambd = deepcopy(initial_condition)
        self.x = None

    def run(self, iterations: int = 1000, stepsize: Union[float, Callable] = 0.1, verbose: bool=False, **kwargs) -> np.ndarray:
        """Run the algorithm for a given number of iterations

        Args:
            iterations: Number of iterations. Defaults to 1000.
            stepsize: If a float is given as input, the stepsize is constant. 
                                                         If a function is given, it must take an iteration k as input and output the corresponding stepsize.. Defaults to 0.1.
            verbose: If True print some information during the evolution of the algorithm. Defaults to False.
            
        Raises:
            TypeError: The number of iterations must be an int
            TypeError: The stepsize must be a float or a callable

        Returns:
            return a tuple (x, lambda) with the sequence of primal solutions and dual variables if enable_log=True.
        """
        if not isinstance(iterations, int):
            raise TypeError("The number of iterations must be an int")
        if not (isinstance(stepsize, float) or callable(stepsize)):
            raise TypeError("The stepsize must be a float or a function")

        if self.enable_log:
            # initialize sequence of x
            x_dims = [iterations]
            for dim in self.x_shape:
                x_dims.append(dim)
            self.x_sequence = np.zeros(x_dims)

            # initialize sequence of lambda
            self.lambda_sequence = {}
            for j in self.agent.in_neighbors:
                self.lambda_sequence[j] = np.zeros(x_dims)

        for k in range(iterations):
            # store current lambda
            if self.enable_log:
                for j in self.agent.in_neighbors:
                    self.lambda_sequence[j][k] = self.lambd[j]

            # perform an iteration
            if not isinstance(stepsize, float):
                step = stepsize(k)
            else:
                step = stepsize
            self.iterate_run(stepsize=step, **kwargs)

            # store primal solution
            if self.enable_log:
                self.x_sequence[k] = self.x
            
            if verbose:
                if self.agent.id == 0:
                    print('Iteration {}'.format(k), end="\r")

        if self.enable_log:
            return (self.x_sequence, self.lambda_sequence)
    
    def _update_local_solution(self, x: np.ndarray, x_neigh: dict, stepsize: float, **kwargs):
        """Update the local solution
        
        Args:
            x: current solution
            x_neigh: solutions of neighbors (dictionary)
            stepsize: step-size for update
        """
        for j, x_j in x_neigh.items():
            self.lambd[j] += stepsize * (x - x_j)
        
        self.x = x
    
    def iterate_run(self, stepsize: float, **kwargs):
        """Run a single iterate of the algorithm
        """
        # TODO extend to non mono-dimensional variables

        # exchange dual variables with neighbors
        lambda_neigh = self.agent.neighbors_exchange(self.lambd, dict_neigh=True)
        deltalambda = np.zeros(self.x_shape)

        for j in self.agent.in_neighbors:
            deltalambda += self.lambd[j] - lambda_neigh[j]
        
        # build local problem
        x = Variable(self.x_shape[0])
        obj_function = self.agent.problem.objective_function + deltalambda @ x
        pb = Problem(obj_function, self.agent.problem.constraints)

        # solve problem and save data
        x = pb.solve()

        # exchange primal variables with neighbors
        x_neigh = self.agent.neighbors_exchange(x)

        self._update_local_solution(x, x_neigh, stepsize, **kwargs)
    
    def get_result(self):
        """Return the current value primal solution and dual variable
    
        Returns:
            tuple (primal, dual): value of primal solution (np.ndarray), dual variables (dictionary of np.ndarray)
        """
        return (self.x, self.lambd)