import numpy as np
from typing import Union, Callable
from ..agents import Agent
from .algorithm import Algorithm
from .consensus import PushSumConsensus


class GradientTracking(Algorithm):
    """Gradient Tracking Algorithm [...]_ 

    From the perspective of agent :math:`i` the algorithm works as follows. For :math:`k=0,1,\\dots`

    .. math::

        x_i^{k+1} & = \\sum_{j=1}^N w_{ij} x_j^k - \\alpha d_i^k \\\\
        d_i^{k+1} & = \\sum_{j=1}^N w_{ij} d_j^k - [ \\nabla f_i (x_i^{k+1}) - \\nabla f_i (x_i^k)]


    where :math:`x_i\\in\\mathbb{R}^n` and :math:`d_i\\in\\mathbb{R}^n`. The weight matrix :math:`W=[w_{ij}]` must be doubly-stochastic. Extensions to other class of weight matrices :math:`W` are not currently supported.

    Args:
        agent (Agent): agent to execute the algorithm
        initial_condition (numpy.ndarray): initial condition for :math:`x_i`
        enable_log (bool): True for enabling log

    Attributes:
        agent (Agent): agent to execute the algorithm
        x0 (numpy.ndarray): initial condition
        x (numpy.ndarray): current value of the local solution
        d (numpy.ndarray): current value of the local tracker
        shape (tuple): shape of the variable
        x_neigh (dict): dictionary containing the local solution of the (in-)neighbors
        d_neigh (dict): dictionary containing the local tracker of the (in-)neighbors
        enable_log (bool): True for enabling log
    """

    def __init__(self, agent: Agent, initial_condition: np.ndarray, enable_log: bool = False):
        super(GradientTracking, self).__init__(agent, enable_log)

        self.x0 = initial_condition
        self.x = initial_condition
        # Initialize tracker at x0
        self.d = self.agent.problem.objective_function.subgradient(initial_condition)

        self.shape = self.x.shape

        self.x_neigh = {}
        self.d_neigh = {}

    def _update_local_solution(self, x: np.ndarray, **kwargs):
        """update the local solution

        Args:
            x: new value

        Raises:
            TypeError: Input must be a numpy.ndarray
            ValueError: Incompatible shapes
        """
        if not isinstance(x, np.ndarray):
            raise TypeError("Input must be a numpy.ndarray")
        if x.shape != self. x0.shape:
            raise ValueError("Incompatible shapes")
        self.x = x

    def _update_local_tracker(self, d: np.ndarray, **kwargs):
        """update the local tracker

        Args:
            d: new value

        Raises:
            TypeError: Input must be a numpy.ndarray
            ValueError: Incompatible shapes
        """
        if not isinstance(d, np.ndarray):
            raise TypeError("Input must be a numpy.ndarray")
        if d.shape != self. x0.shape:
            raise ValueError("Incompatible shapes")
        self.d = d

    def iterate_run(self, stepsize: float, **kwargs):
        """Run a single iterate of the gradient tracking algorithm
        """

        data_x = self.agent.neighbors_exchange(self.x)
        for neigh in data_x:
            self.x_neigh[neigh] = data_x[neigh]

        data_d = self.agent.neighbors_exchange(self.d)
        for neigh in data_d:
            self.d_neigh[neigh] = data_d[neigh]

        x_kp = self.agent.in_weights[self.agent.id] * self.x
        d_kp = self.agent.in_weights[self.agent.id] * self.d

        for j in self.agent.in_neighbors:
            x_kp += self.agent.in_weights[j] * self.x_neigh[j]
            d_kp += self.agent.in_weights[j] * self.d_neigh[j]

        x_kp += - stepsize * self.d
        d_kp += self.agent.problem.objective_function.subgradient(
            x_kp) - self.agent.problem.objective_function.subgradient(self.x)

        self._update_local_solution(x_kp, **kwargs)
        self._update_local_tracker(d_kp, **kwargs)

    def run(self, iterations: int = 1000, stepsize: Union[float, Callable] = 0.1, verbose: bool=False) -> np.ndarray:
        """Run the gradient tracking algorithm for a given number of iterations

        Args:
            iterations: Number of iterations. Defaults to 1000.
            stepsize: If a float is given as input, the stepsize is constant. 
                                                        Default is 0.01.
            verbose: If True print some information during the evolution of the algorithm. Defaults to False.

        Raises:
            TypeError: The number of iterations must be an int
            TypeError: The stepsize must be a float

        Returns:
            return the sequence of estimates if enable_log=True.
        """
        if not isinstance(iterations, int):
            raise TypeError("The number of iterations must be an int")
        if not (isinstance(stepsize, float) or callable(stepsize)):
            raise TypeError("The stepsize must be a float or a function")

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
            self.iterate_run(stepsize=step)

            if self.enable_log:
                self.sequence[k] = self.x
            
            if verbose:
                if self.agent.id == 0:
                    print('Iteration {}'.format(k), end="\r")

        if self.enable_log:
            return self.sequence

    def get_result(self):
        """Return the actual value of x

        Returns:
            numpy.ndarray: value of x
        """
        return self.x


class DirectedGradientTracking(PushSumConsensus):
    """Gradient Tracking Algorithm [XiKh18]_ 

    From the perspective of agent :math:`i` the algorithm works as follows. For :math:`k=0,1,\\dots`

    .. math::

        x_i^{k+1} & = \\sum_{j=1}^N a_{ij} x_j^k - \\alpha y_i^k \\\\
        y_i^{k+1} & = \\sum_{j=1}^N b_{ij} (y_j^k - [ \\nabla f_j (x_j^{k+1}) - \\nabla f_j (x_j^k)])


    where :math:`x_i\\in\\mathbb{R}^n` and :math:`y_i\\in\\mathbb{R}^n`. The weight matrix :math:`A=[a_{ij}]` must be row-stochastic, while :math:`B=[b_{ij}]` must be column-stochastic. The underlying graph can be directed (and unbalanced).
    """


    def __init__(self, agent: Agent, initial_condition: np.ndarray, enable_log: bool = False):
        super(DirectedGradientTracking, self).__init__(agent, initial_condition, enable_log)

        # initialize tracker
        self.y = self.agent.problem.objective_function.subgradient(initial_condition)


    def iterate_run(self, **kwargs):
        """Run a single iterate of the algorithm
        """
        stepsize = kwargs.get('stepsize', 0.1)
        # in
        data = self.agent.neighbors_exchange(self.x)

        for neigh in data:
            self.x_neigh[neigh] = data[neigh]

        x_avg = self.agent.in_weights[self.agent.id] * self.x
        for i in self.agent.in_neighbors:
            x_avg += self.agent.in_weights[i] * self.x_neigh[i]

        subg_old = self.agent.problem.objective_function.subgradient(self.x)
        self._update_x_average(x_avg - stepsize * self.y, **kwargs)
        subg_new = self.agent.problem.objective_function.subgradient(self.x)

        # y average
        send_data = {}
        gradient_diff = subg_new - subg_old
        for j in self.agent.out_neighbors:
            send_data[j] = self.agent.out_weights[j] * (self.y + gradient_diff)
        data = self.agent.neighbors_exchange(send_data, dict_neigh=True)

        for neigh in data:
            self.y_neigh[neigh] = data[neigh]

        y_avg = self.agent.out_weights[self.agent.id] * (self.y + gradient_diff)
        for i in self.agent.in_neighbors:
            y_avg += self.y_neigh[i]

        self._update_y_average(y_avg, **kwargs)

    def get_result(self):
        """Return the actual value of x

        Returns:
            numpy.ndarray: value of x
        """
        return self.x
