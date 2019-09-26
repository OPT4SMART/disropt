import numpy as np
import time
import warnings
from copy import deepcopy
from typing import Union, Callable, Tuple
from ..agents import Agent
from ..constraints.projection_sets import AbstractSet
from .consensus import Consensus, BlockConsensus


class SubgradientMethod(Consensus):
    """Distributed projected (sub)gradient method.

    From the perspective of agent :math:`i` the algorithm works as follows. For :math:`k=0,1,\\dots`

    .. math::

        y_i^{k} &= \sum_{j=1}^N w_{ij} x_j^k

       x_i^{k+1} &= \Pi_{X}[y_i^k - \\alpha^k \\tilde{\\nabla} f_i(y_i^k)]

    where :math:`x_i,y_i\\in\\mathbb{R}^n`, :math:`X\\subseteq\mathbb{R}^n`, :math:`\\alpha^k` is a positive stepsize, :math:`w_{ij}` denotes the weight assigned by agent :math:`i` to agent :math:`j`, :math:`\\Pi_X[]` denotes the projection operator over the set :math:`X` and :math:`\\tilde{\\nabla} f_i(y_i^k)\\in\\partial f_i(y_i^k)` a (sub)gradient of :math:`f_i` computed at :math:`y_i^k`.
    The weight matrix :math:`W=[w_{ij}]_{i,j=1}^N` should be doubly stochastic.

    The algorithm, as written above, was originally presented in [NeOz09]_. Many other variants and extension has been proposed, allowing for stochastic objective functions, time-varying graphs, local stepsize sequences. All these variant can be implemented through the :class:`SubgradientMethod` class.
    """

    def __init__(self, agent: Agent, initial_condition: np.ndarray, enable_log: bool = False):
        if not isinstance(agent, Agent):
            raise TypeError("Agent must be an Agent")
        super(SubgradientMethod, self).__init__(agent, initial_condition, enable_log)

    def _update_local_solution(self, x: np.ndarray, stepsize: float = 0.1, projection: bool = False):
        y = x
        # Perform a sugradient step
        self.x = y - stepsize * self.agent.problem.objective_function.subgradient(y)
        if projection:
            self.x = self.agent.problem.project_on_constraint_set(self.x)

    def run(self, iterations: int = 1000, stepsize: Union[float, Callable] = 0.001, verbose: bool=False) -> np.ndarray:
        """Run the algorithm for a given number of iterations

        Args:
            iterations: Number of iterations. Defaults to 1000.
            stepsize: If a float is given as input, the stepsize is constant. 
                                                         If a function is given, it must take an iteration k as input and output the corresponding stepsize.. Defaults to 0.1.
            verbose: If True print some information during the evolution of the algorithm. Defaults to False.

        Raises:
            TypeError: The number of iterations must be an int
            TypeError: The stepsize must be a float or a callable
            ValueError: Only sets (children of AstractSet) with explicit projections are currently supported
            ValueError: Only one constraint per time is currently supported

        Returns:
            return the sequence of estimates if enable_log=True.
        """
        if not isinstance(iterations, int):
            raise TypeError("The number of iterations must be an int")
        if not (isinstance(stepsize, float) or callable(stepsize)):
            raise TypeError("The stepsize must be a float or a function")

        actv_projection = False
        if len(self.agent.problem.constraints) != 0:
            actv_projection = True

        if self.enable_log:
            dims = [iterations]
            for dim in self.x.shape:
                dims.append(dim)
            self.sequence = np.zeros(dims)

        for k in range(iterations):
            if not isinstance(stepsize, float):
                step = stepsize(k)
            else:
                step = stepsize
            self.iterate_run(stepsize=step, projection=actv_projection)

            if self.enable_log:
                self.sequence[k] = self.x
            
            if verbose:
                if self.agent.id == 0:
                    print('Iteration {}'.format(k), end="\r")

        if self.enable_log:
            return self.sequence


class BlockSubgradientMethod(BlockConsensus):
    """Distributed block subgradient method. This is a special instance of the Block Proximal Method in [FaNo19]_

    At each iteration, the agent can update its local estimate or not at each iteration according to a certain probability (awakening_probability).
    From the perspective of agent :math:`i` the algorithm works as follows. At iteration :math:`k` if the agent is awake, it selects a random block :math:`\\ell_i^k` of its local solution and updates

    .. math::

        \\begin{align}
        y_i^k &= \\sum_{j\\in\\mathcal{N}_i} w_{ij} x_{j\\mid i}^k \\\\
        x_{i,\\ell}^{k+1} &= \\begin{cases}
             \\Pi_{X_\\ell}\\left[y_i^k - \\alpha_i^k [\\tilde{\\nabla} f_i(y_i^k)]_{\\ell}\\right] & \\text{if } \\ell = \\ell_i^k \\\\
             x_{i,\\ell}^{k} & \\text{otherwise}
             \\end{cases}
        \\end{align}

    then it broadcasts :math:`x_{i,\\ell_i^k}^{k+1}` to its out-neighbors. Otherwise (if the agent is not awake) :math:`x_{i}^{k+1}=x_i^k`.
    Here :math:`\\mathcal{N}_i` is the current set of in-neighbors and :math:`x_{j\\mid i},j\\in\\mathcal{N}_i` is the local copy of :math:`x_j` available at node :math:`i` and :math:`x_{i,\\ell}` denotes the :math:`\\ell`-th block of :math:`x_i`. 
    The weight matrix :math:`W=[w_{ij}]_{i,j=1}^N` should be doubly stochastic.

    Notice that if there is only one block and awakening_probability=1 the :class:`BlockSubgradientMethod` reduces to the :class:`SubgradientMethod`.
    """

    def __init__(self, agent: Agent, initial_condition: np.ndarray, **kwargs):
        if not isinstance(agent, Agent):
            raise TypeError("Agent must be an Agent")
        super(BlockSubgradientMethod, self).__init__(agent, initial_condition, **kwargs)

    def _update_local_solution(self, x: np.ndarray, selected_block: Tuple, stepsize: float = 0.1, projection: bool = False):
        block = list(selected_block)
        y = x
        # Perform a sugradient step
        self.x[block] = y[block] - stepsize * self.agent.problem.objective_function.subgradient(y)[block]
        if projection:
            # TODO: single block projection with multiple sets
            old_x = deepcopy(self.x)
            self.x = self.agent.problem.project_on_constraint_set(self.x)
            for idx, value in enumerate(self.x):
                if idx not in block:
                    if old_x[idx] != value:
                        warnings.warn("Constraints are not separable on blocks. Convergence is not guaranteed.")
                        break

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
            ValueError: Only sets (children of AstractSet) with explicit projections are currently supported
            ValueError: Only one constraint per time is currently supported

        Returns:
            return the sequence of estimates if enable_log=True.
        """
        if not isinstance(iterations, int):
            raise TypeError("The number of iterations must be an int")
        if not (isinstance(stepsize, float) or callable(stepsize)):
            raise TypeError("The stepsize must be a float or a function")

        actv_projection = False
        if len(self.agent.problem.constraints) != 0:
            actv_projection = True

        if self.enable_log:
            dims = [iterations]
            for dim in self.x.shape:
                dims.append(dim)
            self.sequence = np.zeros(dims)

        for k in range(iterations):
            if not isinstance(stepsize, float):
                step = stepsize(k)
            else:
                step = stepsize
            self.iterate_run(stepsize=step, projection=actv_projection)

            if self.enable_log:
                self.sequence[k] = self.x
            
            if verbose:
                if self.agent.id == 0:
                    print('Iteration {}'.format(k), end="\r")

        if self.enable_log:
            return self.sequence


