import numpy as np
import time
import warnings
import random
from typing import List, Tuple
from ..agents import Agent
from .algorithm import Algorithm


class Consensus(Algorithm):
    """Consensus Algorithm [OlSa07]_ 

    From the perspective of agent :math:`i` the algorithm works as follows. For :math:`k=0,1,\\dots`

    .. math::

        x_i^{k+1} = \sum_{j=1}^N w_{ij} x_j^k

    where :math:`x_i\\in\\mathbb{R}^n`. The weight matrix :math:`W=[w_{ij}]` should be doubly-stochastic in order to have convergence to the average of the local initial conditions. If :math:`W` is row-stochastic convergence is still attained but at a different point. Other type of matrices can be used, but convergence is not guaranteed.
    Also time-varying graphs can be adopted.

    Args:
        agent (Agent): agent to execute the algorithm
        initial_condition (numpy.ndarray): initial condition
        enable_log (bool): True for enabling log

    Attributes:
        agent (Agent): agent to execute the algorithm
        x0 (numpy.ndarray): initial condition
        x (numpy.ndarray): current value of the local solution
        shape (tuple): shape of the variable
        x_neigh (dict): dictionary containing the local solution of the (in-)neighbors
        enable_log (bool): True for enabling log
    """

    def __init__(self, agent: Agent, initial_condition: np.ndarray, enable_log: bool=False):
        super(Consensus, self).__init__(agent, enable_log)

        self.x0 = initial_condition
        self.x = initial_condition

        self.shape = self.x.shape

        self.x_neigh = {}

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

    def iterate_run(self, **kwargs):
        """Run a single iterate of the algorithm
        """
        data = self.agent.neighbors_exchange(self.x)

        for neigh in data:
            self.x_neigh[neigh] = data[neigh]

        x_avg = self.agent.in_weights[self.agent.id] * self.x
        for i in self.agent.in_neighbors:
            x_avg += self.agent.in_weights[i] * self.x_neigh[i]

        self._update_local_solution(x_avg, **kwargs)

    def run(self, iterations: int=100, verbose: bool=False, **kwargs):
        """Run the algorithm for a given number of iterations

        Args:
            iterations: Number of iterations. Defaults to 100.
            verbose: If True print some information during the evolution of the algorithm. Defaults to False.
        """

        if not isinstance(iterations, int):
            raise TypeError("iterations must be an int")
        if self.enable_log:
            dims = [iterations]
            for dim in self.x.shape:
                dims.append(dim)
            self.sequence = np.zeros(dims)

        for k in range(iterations):
            self.iterate_run(**kwargs)

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


class AsynchronousConsensus(Algorithm):
    """ Asynchronous Consensus Algorithm 

    From the perspective of agent :math:`i` the algorithm works as follows. When agent :math:`i` gets awake it updates its local solution as

    .. math::

        x_i \\gets \sum_{j\\in\\mathcal{N}_i} w_{ij} x_{j\\mid i}

    where :math:`\\mathcal{N}_i` is the current set of in-neighbors and :math:`x_{j\\mid i},j\\in\\mathcal{N}_i` is the local copy of :math:`x_j` available at node :math:`i` (which can be outdated, due to asynchrony, computation time and link failures).

    Args:
        agent: agent to execute the algorithm
        initial_condition: initial condition
        enable_log: True for enabling log. Defaults to False.
        force_sleep: True if one wanst to force sleep after the computation phase. Defaults to False.
        maximum_sleep: Maximum allowed sleep. Defaults to 0.01.
        sleep_type: Type of sleep time("constant", "random"). Defaults to "random".
        force_computation_time:  True if one want sto force length computation phase. Defaults to False.
        maximum_computation_time: Maximum allowed computation time. Defaults to 0.01.
        computation_time_type: Type of computation time ("constant", "random"). Defaults to "random".
        force_unreliable_links: True if one wants to force unreliable links. Defaults to False.
        link_failure_probability: Probability of incoming links failure. Defaults to 0.

    Attributes:
        agent (Agent): agent to execute the algorithm
        x0 (numpy.ndarray): initial condition
        x (numpy.ndarray): current value of the local solution
        shape (tuple): shape of the variable
        x_neigh (dict): dictionary containing the local solution of the (in-)neighbors
        enable_log (bool): True for enabling log
        timestamp_sequence_awake ( list ): list of timestamps at which node get awake
        timestamp_sequence_sleep ( list ): list of timestamps at which node go to sleep
        force_sleep: True if one wanst to force sleep after the computation phase. Defaults to False.
        maximum_sleep: Maximum allowed sleep. Defaults to 0.01.
        sleep_type: Type of sleep time("constant", "random"). Defaults to "random".
        force_computation_time:  True if one want sto force length computation phase. Defaults to False.
        maximum_computation_time: Maximum allowed computation time. Defaults to 0.01.
        computation_time_type: Type of computation time ("constant", "random"). Defaults to "random".
        force_unreliable_links: True if one wants to force unreliable links. Defaults to False.
        link_failure_probability: Probability of incoming links failure. Defaults to 0.
    """

    def __init__(self,
                 agent: Agent,
                 initial_condition: np.ndarray,
                 enable_log: bool = False,
                 force_sleep: bool = False,
                 maximum_sleep: float = 0.01,
                 sleep_type: str = "random",
                 force_computation_time: bool = False,
                 maximum_computation_time: float = 0.01,
                 computation_time_type: str = "random",
                 force_unreliable_links: bool = False,
                 link_failure_probability: float = 0):

        super(AsynchronousConsensus, self).__init__(agent, enable_log)

        self.x0 = initial_condition
        self.x = initial_condition

        self.shape = self.x.shape

        self.x_neigh = {}

        if force_sleep:
            if sleep_type not in ("constant", "random"):
                raise ValueError("sleep_type can be constant or random")
        if force_computation_time:
            if computation_time_type not in ("constant", "random"):
                raise ValueError("computation_time_type can be constant or random")
        self.enable_log = enable_log
        self.force_sleep = force_sleep
        self.maximum_sleep = maximum_sleep
        self.sleep_type = sleep_type
        self.force_computation_time = force_computation_time
        self.maximum_computation_time = maximum_computation_time
        self.computation_time_type = computation_time_type
        self.force_unreliable_links = force_unreliable_links
        self.link_failure_probability = link_failure_probability

        self.timestamp_sequence_awake = None
        self.timestamp_sequence_sleep = None

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

    def iterate_run(self, **kwargs):
        """Run a single iterate
        """

        data = self.agent.neighbors_receive_asynchronous()

        if self.enable_log:
            self.timestamp_sequence_awake.append(time.time())

        for neigh in data:
            self.x_neigh[neigh] = data[neigh]

        if self.force_computation_time:
            start_wait_time = time.time()

        if not self.force_unreliable_links:
            x_avg = self.agent.in_weights[self.agent.id] * self.x
            for i in self.agent.in_neighbors:
                x_avg += self.agent.in_weights[i] * self.x_neigh[i]
        else:
            warnings.warn("when forcing link failures, all neighbors are given the same weight")
            neighbors = []
            for neigh in self.agent.in_neighbors:
                rnd = random.uniform(0, 1)
                if rnd > self.link_failure_probability:
                    neighbors.append(neigh)
            weight = 1.0/(len(neighbors) + 1)
            x_avg = weight * self.x
            for i in neighbors:
                x_avg += weight * self.x_neigh[i]

        self._update_local_solution(x_avg, **kwargs)

        # save sequence
        if self.enable_log:
            self.sequence = np.vstack([self.sequence, self.x.reshape(self.dims)])

        # force computation time if requested
        if self.force_computation_time:
            if self.computation_time_type == "random":
                wait_time = random.uniform(0, self.maximum_computation_time)
            elif self.computation_time_type == "constant":
                wait_time = self.maximum_computation_time
            remaining_wait_time = wait_time - (time.time() - start_wait_time)
            if remaining_wait_time < 0:
                warnings.warn("requested computation time cannot be guaranteed")
            else:
                time.sleep(remaining_wait_time)

        self.agent.neighbors_send(self.x)
        if self.enable_log:
            self.timestamp_sequence_sleep.append(time.time())

    def run(self, running_time: float = 5.0):
        """Run the asynchronous consensus algorithm for a certain amount of time

        Args:
            running_time: Total run time. Defaults to 5.0.

        Returns:
            tuple: timestamp_sequence_awake, timestamp_sequence_sleep, sequence
        """
        if not isinstance(running_time, (int, float)):
            raise TypeError("Running time must be a float")
        if self.enable_log:
            dims = [1]
            for dim in self.x.shape:
                dims.append(dim)
            self.dims = dims
            self.sequence = np.zeros(dims)
            self.sequence[0] = self.x
            self.timestamp_sequence_awake = [time.time()]
            self.timestamp_sequence_sleep = [time.time()]

        # Exchange all data at the beginning
        data = self.agent.neighbors_exchange(self.x)
        for neigh in data:
            self.x_neigh[neigh] = data[neigh]

        # Then go asyncrhronous
        start_time = time.time()
        end_time = start_time + running_time
        while time.time() < end_time:
            self.iterate_run()

            # force a sleep time if requested
            if self.force_sleep:
                start_sleep_time = time.time()
                if self.sleep_type == "random":
                    sleep_time = random.uniform(0, self.maximum_sleep)
                elif self.sleep_type == "constant":
                    sleep_time = self.maximum_sleep

                remaining_sleep_time = sleep_time - \
                    (time.time() - start_sleep_time)
                if remaining_sleep_time < 0:
                    warnings.warn("requested delay cannot be guaranteed")
                else:
                    time.sleep(remaining_sleep_time)

        if self.enable_log:
            return self.timestamp_sequence_awake, \
                self.timestamp_sequence_sleep, \
                self.sequence

    def get_result(self):
        """Return the actual value of x

        Returns:
            numpy.ndarray: value of x
        """
        return self.x


class BlockConsensus(Algorithm):
    """Block-wise consensus [FaNo19]_

    At each iteration, the agent can update its local estimate or not at each iteration according to a certain probability (awakening_probability).
    From the perspective of agent :math:`i` the algorithm works as follows. At iteration :math:`k` if the agent is awake, it selects a random block :math:`\\ell_i^k` of its local solution and updates

    .. math::

        x_{i,\\ell}^{k+1} = \\begin{cases}
             \\sum_{j\\in\\mathcal{N}_i} w_{ij} x_{j\\mid i,\\ell}^k & \\text{if} \\ell = \\ell_i^k \\\\
             x_{i,\\ell}^{k} & \\text{otherwise}
             \\end{cases}

    where :math:`\\mathcal{N}_i` is the current set of in-neighbors and :math:`x_{j\\mid i},j\\in\\mathcal{N}_i` is the local copy of :math:`x_j` available at node :math:`i` and :math:`x_{i,\\ell}` denotes the :math:`\\ell`-th block of :math:`x_i`. Otherwise :math:`x_{i}^{k+1}=x_i^k`.



    Args:
        agent: agent to execute the algorithm
        initial_condition: initial condition
        enable_log: True for enabling log
        blocks_list: the list of blocks (list of tuples)
        probabilities: list of probabilities of drawing each block
        awakening_probability: probability of getting awake at each iteration

    """

    def __init__(self,
                 agent: Agent,
                 initial_condition: np.ndarray,
                 enable_log: bool = False,
                 blocks_list: List[Tuple] = None,
                 probabilities: List[float] = None,
                 awakening_probability: float = 1.0):

        super(BlockConsensus, self).__init__(agent, enable_log)
        if (not isinstance(initial_condition, np.ndarray)) or \
            (len(initial_condition.shape) != 2) or \
                (initial_condition.shape[1] != 1):
            raise ValueError("Initial condition must be a numpy.ndarray with shape (Any, 1)")

        self.x0 = initial_condition
        self.x = initial_condition

        self.shape = self.x.shape

        self.x_neigh = {}
        if blocks_list is not None:
            if not isinstance(blocks_list, list):
                raise ValueError("blocks_list argument, if provided, must be a list of tuples")
            items = 0
            for item in blocks_list:
                if not isinstance(item, tuple):
                    raise ValueError("blocks_list argument, if provided, must be a list of tuples")
                items += len(item)
            self.blocks_list = blocks_list
            if items != self.shape[0]:
                warnings.warn("Not all elements have been included in blocks_list")
        else:
            self.blocks_list = list(range(self.shape[0]))

        self.blocks_number = len(self.blocks_list)

        if probabilities is not None:
            if isinstance(probabilities, list):
                for i in probabilities:
                    if not isinstance(i, float):
                        raise ValueError("probabilities argument, if provided, must be a list of float")
            else:
                raise ValueError("probabilities argument, if provided, must be a list of float")

            if len(probabilities) != len(self.blocks_list):
                raise ValueError("blocks_list and probabilities arguments have different lengths")

            if sum(probabilities) != 1.0:
                raise ValueError("probabilities must sum to 1")

            self.probabilities = probabilities
        else:
            self.probabilities = (np.ones(self.shape).flatten()/self.shape[0]).tolist()

        if isinstance(awakening_probability, float) and (0 <= awakening_probability <= 1):
            self.awakening_probability = awakening_probability
        else:
            raise ValueError("awakening_probability must be a float in [0,1]")

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

    def iterate_run(self, **kwargs):
        """Run a single iterate of the algorithm
        """
        awake = random.uniform(0, 1)
        if awake <= self.awakening_probability:
            selected_index = np.random.choice(np.arange(self.blocks_number), p=self.probabilities)
            
            selected_block = self.blocks_list[selected_index]
            if isinstance(selected_block, int):
                selected_block = (selected_block, )
            packet_send = {'block': selected_block, 'data': self.x[list(selected_block)]}
            data = self.agent.neighbors_exchange(packet_send)

            for neigh in data:
                received_block = data[neigh]['block']
                received_data = data[neigh]['data']
                try:
                    self.x_neigh[neigh][list(received_block)] = received_data
                except KeyError:
                    self.x_neigh[neigh] = np.zeros(self.shape)
                    self.x_neigh[neigh][list(received_block)] = received_data

            x_avg = self.agent.in_weights[self.agent.id] * self.x
            for i in self.agent.in_neighbors:
                x_avg += self.agent.in_weights[i] * self.x_neigh[i]

            self._update_local_solution(x_avg, selected_block=selected_block, **kwargs)

    def run(self, iterations: int=100, verbose: bool=False):
        """Run the algorithm for a given number of iterations

        Args:
            iterations: Number of iterations. Defaults to 100.
            verbose: If True print some information during the evolution of the algorithm. Defaults to False.
        """
        if not isinstance(iterations, int):
            raise TypeError("iterations must be an int")
        if self.enable_log:
            dims = [iterations]
            for dim in self.x.shape:
                dims.append(dim)
            self.sequence = np.zeros(dims)

        for k in range(iterations):
            self.iterate_run()

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


class PushSumConsensus(Algorithm):
    """Push-Sum Consensus Algorithm 

    From the perspective of agent :math:`i` the algorithm works as follows. For :math:`k=0,1,\\dots`

    .. math::

        x_i^{k+1} &= \\sum_{j=1}^N w_{ij} x_j^k

        y_i^{k+1} &= \\sum_{j=1}^N w_{ij} y_j^k

        z_i^{k+1} &= \\frac{x_i^{k+1}}{y_i^{k+1}}

    where :math:`x_i\\in\\mathbb{R}^n`. The weight matrix :math:`W=[w_{ij}]` should be column-stochastic in order to let :math:`z_i^k` converge to the average of the local initial conditions.
    Also time-varying graphs can be adopted.


    Args:
        agent (Agent): agent to execute the algorithm
        initial_condition (numpy.ndarray): initial condition
        enable_log (bool): True for enabling log

    Attributes:
        agent (Agent): agent to execute the algorithm
        z0 (numpy.ndarray): initial condition
        z (numpy.ndarray): current value of the local solution
        shape (tuple): shape of the variable
        x_neigh (dict): dictionary containing the x values of the (in-)neighbors
        y_neigh (dict): dictionary containing the y values of the (in-)neighbors
        enable_log (bool): True for enabling log
    """

    def __init__(self, agent: Agent, initial_condition: np.ndarray, enable_log: bool=False):
        super(PushSumConsensus, self).__init__(agent, enable_log)

        self.z0 = initial_condition
        self.x = initial_condition
        self.z = initial_condition
        self.y = np.ones(initial_condition.shape)

        self.shape = self.z0.shape

        self.x_neigh = {}
        self.y_neigh = {}

    def _update_x_average(self, x: np.ndarray, **kwargs):
        if not isinstance(x, np.ndarray):
            raise TypeError("Input must be a numpy.ndarray")
        if x.shape != self.x.shape:
            raise ValueError("Incompatible shapes")
        self.x = x
    
    def _update_y_average(self, y: np.ndarray, **kwargs):
        if not isinstance(y, np.ndarray):
            raise TypeError("Input must be a numpy.ndarray")
        if y.shape != self.y.shape:
            raise ValueError("Incompatible shapes")
        self.y = y

    def _update_local_solution(self, z: np.ndarray, **kwargs):
        """update the local solution
        
        Args:
            x: new value
        
        Raises:
            TypeError: Input must be a numpy.ndarray
            ValueError: Incompatible shapes
        """
        if not isinstance(z, np.ndarray):
            raise TypeError("Input must be a numpy.ndarray")
        if z.shape != self.z.shape:
            raise ValueError("Incompatible shapes")
        self.z = z

    def iterate_run(self, **kwargs):
        """Run a single iterate of the algorithm
        """
        # # in
        # data = self.agent.neighbors_exchange(self.x)

        # for neigh in data:
        #     self.x_neigh[neigh] = data[neigh]

        # x_avg = self.agent.in_weights[self.agent.id] * self.x
        # for i in self.agent.in_neighbors:
        #     x_avg += self.agent.in_weights[i] * self.x_neigh[i]

        # x average
        send_data = {}
        for j in self.agent.out_neighbors:
            send_data[j] = self.agent.out_weights[j] * self.x
        data = self.agent.neighbors_exchange(send_data, dict_neigh=True)
        
        for neigh in data:
            self.x_neigh[neigh] = data[neigh]

        x_avg = self.agent.out_weights[self.agent.id] * self.x
        for i in self.agent.in_neighbors:
            x_avg += self.x_neigh[i]

        self._update_x_average(x_avg, **kwargs)

        # y average
        send_data = {}
        for j in self.agent.out_neighbors:
            send_data[j] = self.agent.out_weights[j] * self.y
        data = self.agent.neighbors_exchange(send_data, dict_neigh=True)

        for neigh in data:
            self.y_neigh[neigh] = data[neigh]

        y_avg = self.agent.out_weights[self.agent.id] * self.y
        for i in self.agent.in_neighbors:
            y_avg += self.y_neigh[i]

        self._update_y_average(y_avg, **kwargs)

        # aggregate
        z = self.x/self.y
        self._update_local_solution(z)

    def run(self, iterations: int=100, verbose: bool=False, **kwargs):
        """Run the algorithm for a given number of iterations

        Args:
            iterations: Number of iterations. Defaults to 100.
            verbose: If True print some information during the evolution of the algorithm. Defaults to False.
        """
        if not isinstance(iterations, int):
            raise TypeError("iterations must be an int")
        if self.enable_log:
            dims = [iterations]
            for dim in self.x.shape:
                dims.append(dim)
            self.sequence = np.zeros(dims)

        for k in range(iterations):
            self.iterate_run(**kwargs)

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
        return self.z
