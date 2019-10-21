import numpy as np
from copy import deepcopy
from ..agents.agent import Agent
from ..problems.problem import Problem
from .algorithm import Algorithm
from ..functions import Variable
from ..functions.squared_norm import SquaredNorm
from ..functions.quadratic_form import QuadraticForm


class ADMM(Algorithm):
    """Distributed ADMM.

    From the perspective of agent :math:`i` the algorithm works as follows.

    Initialization: :math:`\lambda_{ij}^0` for all :math:`j \in \mathcal{N}_i`, :math:`\lambda_{ii}^0` and :math:`z_i^0`

    For :math:`k=0,1,\\dots`

    * Compute :math:`x_i^k` as the optimal solution of

    .. math::
        \min_{x_i \in X_i} \: f_i(x_i) + x_i^\\top \sum_{j \in \mathcal{N}_i \cup \\{i\\}} \lambda_{ij}^k + \\frac{\\rho}{2} \sum_{j \in \mathcal{N}_i \cup \\{i\\}} \\| x_i - z_j^t \\|^2

    * Gather :math:`x_j^k` and :math:`\lambda_{ji}^k` from neighbors :math:`j \in \mathcal{N}_i`

    * Update :math:`z_i^{k+1}`

    .. math::
        z_i^{k+1} = \\frac{\sum_{j \in \mathcal{N}_i \cup \\{i\\}} x_j^k}{|\mathcal{N}_i| + 1} + \\frac{\sum_{j \in \mathcal{N}_i \cup \\{i\\}} \lambda_{ji}^k}{\\rho (|\mathcal{N}_i| + 1)}

    * Gather :math:`z_j^{k+1}` from neighbors :math:`j \in \mathcal{N}_i`

    * Update for all :math:`j \in \mathcal{N}_i`

    .. math::
        \lambda_{ij}^{k+1} = \lambda_{ij}^{k} + \\rho (x_i^k - z_j^{k+1})

    where :math:`x_i, z_i\\in\\mathbb{R}^{n}`, :math:`\lambda_{ij}\\in\\mathbb{R}^n` for all :math:`j \in \mathcal{N}_i \cup \\{i\\}`, :math:`X_i\\subseteq\mathbb{R}^{n}` for all :math:`i` and :math:`\\rho` is a positive penalty parameter.

    The algorithm has been presented in ????.
    """

    # TODO choose ref
    def __init__(self, agent: Agent, initial_lambda: dict, initial_z: np.ndarray, enable_log: bool = False):
        super(ADMM, self).__init__(agent, enable_log)

        if not isinstance(agent.problem, Problem):
            raise TypeError("The agent must be equipped with a Problem")

        if sum(1 for i in agent.problem.objective_function.input_shape if i > 1) > 1:
            raise ValueError("Currently only mono-dimensional objective functions are supported")

        if not all([isinstance(x, np.ndarray) for _, x in initial_lambda.items()]):
            raise TypeError("The initial condition dictionary can only contain numpy vectors")

        # augmented set of neighbors and degree
        self.augmented_neighbors = deepcopy(self.agent.in_neighbors)
        self.augmented_neighbors.append(agent.id)
        self.augmented_neighbors.sort()
        self.degree = len(agent.in_neighbors)

        if self.augmented_neighbors != sorted([x for x in initial_lambda]):
            raise TypeError(
                "The initial condition dictionary must contain exactly one vector per neighbor (plus the agent itself)")

        # shape of local variable
        self.x_shape = agent.problem.objective_function.input_shape

        # initialize dual variables and primal solution
        self.lambd0 = deepcopy(initial_lambda)
        self.lambd = deepcopy(initial_lambda)
        self.z0 = initial_z
        self.z = initial_z
        self.x = None

    def run(self, iterations: int = 1000, penalty: float = 0.1, verbose: bool = False, **kwargs) -> np.ndarray:
        """Run the algorithm for a given number of iterations

        Args:
            iterations: Number of iterations. Defaults to 1000.
            penalty: ADMM penalty parameter. Defaults to 0.1.
            verbose: If True print some information during the evolution of the algorithm. Defaults to False.

        Raises:
            TypeError: The number of iterations must be an int

        Returns:
            return a tuple (x, lambda, z) with the sequence of primal solutions, dual variables and auxiliary primal variables if enable_log=True.
        """
        if not isinstance(iterations, int):
            raise TypeError("The number of iterations must be an int")

        if self.enable_log:
            # initialize sequence of x and z
            x_dims = [iterations]
            for dim in self.x_shape:
                x_dims.append(dim)
            self.x_sequence = np.zeros(x_dims)
            self.z_sequence = np.zeros(x_dims)

            # initialize sequence of lambda
            self.lambda_sequence = {}
            for j in self.augmented_neighbors:
                self.lambda_sequence[j] = np.zeros(x_dims)

        self.initialize_algorithm()

        for k in range(iterations):
            # store current lambda and z
            if self.enable_log:
                for j in self.augmented_neighbors:
                    self.lambda_sequence[j][k] = self.lambd[j]
                self.z_sequence[k] = self.z

            self.iterate_run(rho=penalty, **kwargs)

            # store primal solution
            if self.enable_log:
                self.x_sequence[k] = self.x

            if verbose:
                if self.agent.id == 0:
                    print('Iteration {}'.format(k), end="\r")

        if self.enable_log:
            return (self.x_sequence, self.lambda_sequence, self.z_sequence)

    def _update_local_solution(self, x: np.ndarray, z: np.ndarray, z_neigh: dict, rho: float, **kwargs):
        """Update the local solution

        Args:
            x: current solution
            z: current auxiliary primal variable
            z_neigh: auxiliary primal variables of neighbors (dictionary)
            rho: penalty parameter
        """
        self.z_neigh = z_neigh

        # update dual variables
        self.lambd[self.agent.id] += rho * (x - z)
        for j, z_j in z_neigh.items():
            self.lambd[j] += rho * (x - z_j)

        # update primal variables
        self.x = x
        self.z = z

    def initialize_algorithm(self):
        """Initializes the algorithm
        """
        # exchange z with neighbors
        self.z_neigh = self.agent.neighbors_exchange(self.z)

    def iterate_run(self, rho: float, **kwargs):
        """Run a single iterate of the algorithm
        """
        # TODO extend to non mono-dimensional variables

        # build local problem
        x_i = Variable(self.x_shape[0])

        penalties = [(x_i - self.z_neigh[j])@(x_i - self.z_neigh[j]) for j in self.agent.in_neighbors]
        penalties.append((x_i - self.z)@(x_i - self.z))
        sumlambda = sum(self.lambd.values())

        obj_function = self.agent.problem.objective_function + sumlambda @ x_i + (rho/2) * sum(penalties)
        pb = Problem(obj_function, self.agent.problem.constraints)

        # solve problem and save data
        x = pb.solve()

        # exchange primal variables and dual variables with neighbors
        x_neigh = self.agent.neighbors_exchange(x)
        lambda_neigh = self.agent.neighbors_exchange(self.lambd, dict_neigh=True)

        # compute auxiliary variable
        z = (sum(x_neigh.values()) + x) / (self.degree+1) + \
            (sum(lambda_neigh.values()) + self.lambd[self.agent.id]) / (rho*(self.degree+1))

        # exchange auxiliary variables with neighbors
        z_neigh = self.agent.neighbors_exchange(z)

        # update local data
        self._update_local_solution(x, z, z_neigh, rho, **kwargs)

    def get_result(self):
        """Return the current value primal solution, dual variable and auxiliary primal variable

        Returns:
            tuple (primal, dual, auxiliary): value of primal solution (np.ndarray), dual variables (dictionary of np.ndarray), auxiliary primal variable (np.ndarray)
        """
        return (self.x, self.lambd, self.z)
