from typing import Union, Any
from ..communicators import Communicator, MPICommunicator
from ..problems import Problem


class Agent():
    """The Agent object represents an agent in a network with communication capabilities

        Args:
            in_neighbors (list): list of agents from which communication is received
            out_neighbors (list): list of agents to which information is send
            communicator (Communicator, optional): a Communicator object used to perform communications (if none is provided, it is automatically set to MPICommunicator). Defaults to None.
            in_weights (list or dict, optional): list or dict containing weights to assign to information coming from each in-neighbor. If a list is provided, it must have lenght equal to the number of agents in the network. If a dict is provided, it must have a key for each in-neighbor and, associated to it, the correspondig weight. Defaults to None, implies equal in_weights to in-neighbors.
            out_weights (list or dict, optional): list or dict containing weights to assign to out-neighbor. If a list is provided, it must have lenght equal to the number of agents in the network. If a dict is provided, it must have a key for each out-neighbor and, associated to it, the correspondig weight. Defaults to None, implies equal in_weights to out-neighbors.
            auto_local (bool, optional): If False the (in-)weight for the local agent must be provided. Otherwise it is set automatically, provided that the in_weights have sum in [0,1]. Defaults to True.

        Attributes:
            id (int): id of the Agent
            in_neighbors (list): list of in-neighbors
            out_neighbors (list): list of out-neighbors
            in_weights (dict): a dict containing weights to assign to information coming from each in-neighbor.
            out_weights (dict): a dict containing weights to assign to out-neighbors.
            communicator (Communicator): Communicator object used to perform communications.
            problem (Problem): Local optimization problem.
        """

    def __init__(self, in_neighbors: list = None, out_neighbors: list = None, communicator: Communicator = None,
                 in_weights: Union[list, dict] = None, out_weights: Union[list, dict] = None, auto_local: bool = True):

        if communicator is not None:
            if isinstance(communicator, Communicator):
                self.communicator = communicator
            else:
                raise ValueError(
                    "the communicator must be and instance of the Communicator class.")
        else:
            self.communicator = MPICommunicator()

        self.id = self.communicator.rank

        if in_neighbors is None:
            in_neighbors = list(range(self.communicator.size)).pop(self.id)

        if out_neighbors is None:
            out_neighbors = in_neighbors

        # Set neighbors
        self.in_neighbors = None
        self.out_neighbors = None
        self.set_neighbors(in_neighbors, out_neighbors)

        self.in_weights = {}
        self.set_weights(in_weights, out_weights, auto_local)

        self.problem = None

    def set_neighbors(self, in_neighbors: list, out_neighbors: list):
        """Set in and out neighbors

        Args:
            in_neighbors: list of agents from which communication is received
            out_neighbors: list of agents to which information is send
        """

        if isinstance(in_neighbors, list) and isinstance(out_neighbors, list):
            self.in_neighbors = in_neighbors
            self.out_neighbors = out_neighbors

            # Remove self-loops if any
            if self.id in self.in_neighbors:
                self.in_neighbors.pop(self.id)
            if self.id in self.out_neighbors:
                self.out_neighbors.pop(self.id)

    def set_weights(self, in_weights: Union[list, dict] = None, out_weights: Union[list, dict] = None, auto_local: bool = True):
        """Set in_weights to assign to in-neighbors and the one for agent itself.

        Args:
            in_weights: list or dict contatining in_weights to assign to information coming from each in-neighbor. If a list is provided, it must have lenght equal to the number of agents in the network. If a dict is provided, it must have a key for each in-neighbor and, associated to it, the correspondig weight. Defaults to None, implies equal in_weights to in-neighbors.
            out_weights: list or dict contatining in_weights to assign to out-neighbors. If a list is provided, it must have lenght equal to the number of agents in the network. If a dict is provided, it must have a key for each out-neighbor and, associated to it, the correspondig weight. Defaults to None, implies equal in_weights to out neighbors.
            auto_local: If False the weight for the local agent must be provided. Otherwise it is set automatically, provided that the in_weights have sum in [0,1]. Defaults to True.
        Raises:
            ValueError: If a dict is provided as argument, it must contain a key for each in-neighbor.
            ValueError: Input must be list or dict
            ValueError: If auto_local is not False, the provided in_weights must have sum in [0,1]
        """
        self.in_weights = {}
        self.out_weights = {}
        if in_weights is not None:
            # if a list is provided with length equal to the number of agents in the network
            # it is converted in a dict
            if isinstance(in_weights, list) and len(in_weights) == self.communicator.size:
                for neighbor in self.in_neighbors:
                    self.in_weights[neighbor] = in_weights[neighbor]

            # if a dict is provided it is checked if all in neighbors have a weight
            else:
                if isinstance(in_weights, dict):
                    for neighbor in self.in_neighbors:
                        if neighbor in in_weights:
                            self.in_weights[neighbor] = in_weights[neighbor]
                        else:
                            raise ValueError(
                                "A weight for each in-neighbor must be provided.")
                else:
                    raise ValueError("The in_weights argument must be a list or a dict. \
                            If a list is provided, it must have lenght equal to the number of agents in the network. \
                            If a dict is provided, it must have a key for each in-neighbor and, associated to it, the correspondig weight.")

            # Assign self weight (if sum of provided in_weights in [0,1])
            if not auto_local:
                self.in_weights[self.id] = in_weights[self.id]
            else:
                if 0 <= sum(self.in_weights.values()) <= 1:
                    self.in_weights[self.id] = 1 - sum(self.in_weights.values())
                else:
                    raise ValueError(
                        "If auto_local is set to True, the provided in_weights must have sum in [0,1]")
        else:
            for neighbor in self.in_neighbors:
                self.in_weights[neighbor] = 1/len(self.in_neighbors)

        if out_weights is not None:
            # if a list is provided with length equal to the number of agents in the network
            # it is converted in a dict
            if isinstance(out_weights, list) and len(out_weights) == self.communicator.size:
                for neighbor in self.out_neighbors:
                    self.out_weights[neighbor] = out_weights[neighbor]

            # if a dict is provided it is checked if all in neighbors have a weight
            else:
                if isinstance(out_weights, dict):
                    for neighbor in self.out_neighbors:
                        if neighbor in out_weights:
                            self.out_weights[neighbor] = out_weights[neighbor]
                        else:
                            raise ValueError(
                                "A weight for each in-neighbor must be provided.")
                else:
                    raise ValueError("The in_weights argument must be a list or a dict. \
                            If a list is provided, it must have lenght equal to the number of agents in the network. \
                            If a dict is provided, it must have a key for each in-neighbor and, associated to it, the correspondig weight.")
        
            # Assign self out_weight (if sum of provided in_weights in [0,1])
            if not auto_local:
                self.out_weights[self.id] = out_weights[self.id]
            else:
                if 0 <= sum(self.out_weights.values()) <= 1:
                    self.out_weights[self.id] = 1 - sum(self.out_weights.values())
                else:
                    raise ValueError(
                        "If auto_local is set to True, the provided out_weights must have sum in [0,1]")
        else:
            for neighbor in self.out_neighbors:
                self.out_weights[neighbor] = 1/len(self.out_neighbors)

    def neighbors_exchange(self, obj: Any, dict_neigh=False):
        """Exchange data with neighbors (synchronously). Send obj to the out-neighbors and receive received_obj from in-neighbors

        Args:
            obj: object to send
            dict_neigh: True if obj contains a dictionary with different objects for each neighbor. Defaults to False.

        Returns:
            dict: a dictionary containing an object for each in-neighbor
        """
        received_obj = self.communicator.neighbors_exchange(
            obj, self.in_neighbors, self.out_neighbors, dict_neigh)
        return received_obj

    def neighbors_send(self, obj: Any):
        """Send data to out-neighbors

        Args:
            obj: object to send
        """
        self.communicator.neighbors_send(obj, self.out_neighbors)

    def neighbors_receive_asynchronous(self):
        """Receive data from in-neighbors (if any have been sent)

        Returns:
            dict: a dictionary containing an object for each in-neighbor that has sent one
        """
        received_obj = self.communicator.neighbors_receive_asynchronous(
            self.out_neighbors)
        return received_obj

    def set_problem(self, problem: Problem):
        """set the local optimization problem

        Args:
            problem (Problem): Problem object

        Raises:
            TypeError: Input must be a Problem object
        """
        if not isinstance(problem, Problem):
            raise TypeError("Input must be a Problem object")
        self.problem = problem

