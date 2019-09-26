import numpy as np
from typing import Union, Callable
from ..agents.agent import Agent
from .consensus import Consensus
from ..problems.problem import Problem


class DualSubgradientMethod(Consensus):
    """Distributed dual subgradient method.

    From the perspective of agent :math:`i` the algorithm works as follows. For :math:`k=0,1,\\dots`

    .. math::

        y_i^{k} &= \sum_{j=1}^N w_{ij} \lambda_j^k

        x_i^{k+1} &\in \\text{arg} \min_{x_i \in X_i} f_i(x_i) + g_i(x_i)^\\top y_i^k

        \lambda_i^{k+1} &= \Pi_{\lambda \ge 0}[y_i^k + \\alpha^k g_i(x_i^{k+1})]

        \hat{x}_i^{k+1} &= \hat{x}_i^{k} + \\frac{\\alpha^k}{\sum_{r=0}^{k} \\alpha^r} (x_i^{k+1} - \hat{x}_i^{k})

    where :math:`x_i, \hat{x}_i\\in\\mathbb{R}^{n_i}`, :math:`\lambda_i,y_i\\in\\mathbb{R}^S`, :math:`X_i\\subseteq\mathbb{R}^{n_i}` for all :math:`i`, :math:`\\alpha^k` is a positive stepsize and :math:`\\Pi_{\lambda \ge 0}[]` denotes the projection operator over the nonnegative orthant.

    The algorithm has been presented in [FaMa17]_.

    .. warning::
        this algorithm is still under development
    """

    def __init__(self, agent: Agent, initial_condition: np.ndarray, initial_runavg: np.ndarray = None, enable_log: bool = False):
        super(DualSubgradientMethod, self).__init__(agent, initial_condition, enable_log)

        # initalize running average
        if initial_runavg is None:
            self.x_hat = np.zeros(self.agent.problem.objective_function.input_shape)
        else:
            self.x_hat = initial_runavg

        # initialize cumulative sum of stepsize
        self.stepsize_sum = 0
        
        # number of coupling constraints
        self.S = initial_condition.size

    def _update_local_solution(self, x: np.ndarray, stepsize: float = 0.1):
        y = x

        # define lagrangian: f_i(x_i) + y^top g_i(x_i)
        lagrangian = self.agent.problem.objective_function + (y @ self.agent.problem.coupling_function)

        # create local problem
        pb = Problem(lagrangian, self.agent.problem.constraints)

        # solve local problem
        x_lagr = pb.solve()

        # perform a sugradient step on the dual problem
        lambda_t = y + stepsize * self.agent.problem.coupling_function.eval(x_lagr)

        # project onto non-negative orthant
        lambda_t = np.maximum(lambda_t, np.zeros((self.S, 1)))
        self.x = lambda_t

        # update running average
        self.stepsize_sum += stepsize
        self.x_hat = self.x_hat + stepsize/self.stepsize_sum * (x_lagr - self.x_hat)


    def run(self, iterations: int = 1000, stepsize: Union[float, Callable] = 0.1, verbose: bool=False) -> np.ndarray:
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
            return a tuple (lambda, x_hat) with the sequence of dual and primal estimates if enable_log=True.
        """
        if not isinstance(iterations, int):
            raise TypeError("The number of iterations must be an int")
        if not (isinstance(stepsize, float) or callable(stepsize)):
            raise TypeError("The stepsize must be a float or a function")

        if self.enable_log:
            # initialize sequence of lambda
            lambda_dims = [iterations]
            for dim in self.x.shape:
                lambda_dims.append(dim)
            self.lambda_sequence = np.zeros(lambda_dims)

            # initialize sequence of x hat
            xhat_dims = [iterations]
            for dim in self.x_hat.shape:
                xhat_dims.append(dim)
            self.xhat_sequence = np.zeros(xhat_dims)

        for k in range(iterations):
            if not isinstance(stepsize, float):
                step = stepsize(k)
            else:
                step = stepsize
            self.iterate_run(stepsize=step)

            if self.enable_log:
                self.lambda_sequence[k] = self.x
                self.xhat_sequence[k] = self.x_hat
            
            if verbose:
                if self.agent.id == 0:
                    print('Iteration {}'.format(k), end="\r")

        if self.enable_log:
            return (self.lambda_sequence, self.xhat_sequence)
    
    def get_result(self):
        """Return the actual value of dual and primal averaged variable
    
        Returns:
            tuple (dual, primal) of numpy.ndarray: value of primal running average, value of dual variable
        """
        return (self.x, self.x_hat)