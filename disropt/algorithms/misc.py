import numpy as np
import time
from typing import Any 
from ..agents import Agent
from .algorithm import Algorithm


class LogicAnd(Algorithm):
    """Logic-And algorithm. It can be used for checking in a distributed way if a certain condition (corresponding to flag=True in the algorithm) is satisfied by all the agents in the network. Details can be found in [FaGa19a]_
        
    Args:
        agent (Agent): Agent
        graph_diameter (int): diameter of the graph representing the network 
        flag (bool, optional): local flag value. Defaults to False.
        enable_log (bool, optional): True for enabling log. Defaults to False.
    """

    def __init__(self, agent: Agent, graph_diameter: int, flag: bool = False, enable_log: bool = False, **kwargs):
        super(LogicAnd, self).__init__(agent, enable_log, **kwargs)

        self.flag = flag
        self.graph_diameter = graph_diameter
        self.S = np.zeros([graph_diameter, len(self.agent.in_neighbors) + 1])
        self.S_ind = {}
        for j in range(len(self.agent.in_neighbors)):
            self.S_ind[self.agent.in_neighbors[j]] = j

    def change_flag(self, new_flag: bool):
        """Change the local flag

        Args:
            new_flag: new flag
        """
        if not isinstance(new_flag, bool):
            raise TypeError("new flag must be a bool")
        self.flag = new_flag

    def matrix_update(self):
        """Update the matrix S
        """
        self.S[0, -1] = int(self.flag)
        for l in range(self.graph_diameter-1):
            self.S[l+1, -1] = np.prod(self.S[l, :])

    def force_matrix_update(self):
        """
        Force the matrix S to have all ones in the last row
        """
        self.S[-1] = 1

    def update_column(self, neighbor: Any, column: np.ndarray):
        """Update a column of the matrix corresponding to a neighbor
        
        Args:
            neighbor: neighbor
            column: column value
        
        Raises:
            TypeError: second argument must be a numpy.ndarray with shape (graph_diameter, )
            ValueError: second argument must be a numpy.ndarray with shape (graph_diameter, )
        """
        if not isinstance(column, np.ndarray):
            raise TypeError("second argument must be a numpy.ndarray with shape (graph_diameter, )")
        if column.shape != (self.graph_diameter, ):
            raise ValueError("second argument must be a numpy.ndarray with shape (graph_diameter, )")

        index = self.S_ind[neighbor]
        self.S[:, index] = column

    def check_stop(self):
        """Check the last row of S
        
        Returns:
            bool: True if last row contains only ones. Meaning that all have the flag True
        """
        return bool(np.prod(self.S[-1, :]))

    def iterate_run(self):
        """Run an iterate
        """
        data = self.agent.neighbors_exchange(self.S[:, -1])
        for neigh in data:
            self.update_column(neigh, data[neigh])
        self.matrix_update()

    def run(self, maximum_iterations: int = 100, verbose: bool=False):
        """Run the algorithm
        
        Args:
            maximum_iterations: Maximum number of iterations. Defaults to 100.
            verbose: If True print some information during the evolution of the algorithm. Defaults to False.
        
        Raises:
            TypeError: maximum iterations must be an int
        """
        if not isinstance(maximum_iterations, int):
            raise TypeError("maximum iterations must be an int")
        k = 0
        while k < maximum_iterations:
            self.iterate_run()
            if self.check_stop():
                print("logic-and completed in {} iterations".format(k))
                break
            if verbose:
                if self.agent.id == 0:
                    print('Iteration {}'.format(k), end="\r")
            k += 1
    
    def matrix_reset(self):
        self.flag = False
        self.S = np.zeros(self.S.shape)


class AsynchronousLogicAnd(LogicAnd):
    """Asyncrhonous Logic-And algorithm. It can be used for checking in a distributed way if a certain condition (corresponding to flag=True in the algorithm) is satisfied by all the agents in the network. Details can be found in [FaGa19a]_
        
    Args:
        agent (Agent): Agent
        graph_diameter (int): diameter of the graph representing the network 
        flag (bool, optional): local flag value. Defaults to False.
        enable_log (bool, optional): True for enabling log. Defaults to False.
    """
    def iterate_run(self):
        """Run an iterate
        """
        data = self.agent.neighbors_receive_asynchronous()

        for neigh in data:
            self.update_column(neigh, data[neigh])
        self.matrix_update()

        self.agent.neighbors_send(self.S[:, -1])

    def run(self, maximum_running_time: float = 1):
        """Run the algorithm
        
        Args:
            maximum_running_time: Maximum running time. Defaults to 1.
        
        Raises:
            TypeError: maximum running time must be a float
        """
        if not isinstance(maximum_running_time, float):
            raise TypeError("maximum running time must be a float")
        start_time = time.time()
        end_time = start_time + maximum_running_time
        while time.time() <= end_time:
            self.iterate_run()
            if self.check_stop():
                print("logic-and completed in {} s".format(time.time()-start_time))
                break
