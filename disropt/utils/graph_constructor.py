import numpy as np
from mpi4py import MPI


def binomial_random_graph(N: int, p: float=None, seed: int=None, links_type: str='undirected') -> np.ndarray:
    """construct a random binomial graph
    
    Args:
        N (int): number of agents
        p (float, optional): link probability. Defaults to None (=1).
        seed (int, optional): [description]. Defaults to None (=1).
        links_type (str, optional): 'directed' or 'undirected'. Defaults to 'undirected'.
    
    Returns:
        numpy.ndarray: adjacency matrix
    """
    if p is None:
        p = 1
    if seed is None:
        seed = 1
    np.random.seed(seed)

    while True:
        Adj = np.random.binomial(1, p, (N, N))  # each entry is 1 with prob "p"

        if links_type == 'undirected':
            Adj = np.logical_or(Adj, Adj.transpose())  # Undirected
        I_NN = np.eye(N)

        # Adj = np.logical_or(Adj, I_NN).astype(int)  # add self - loops
        Adj = np.logical_and(Adj, np.logical_not(I_NN)).astype(int)  # remove self - loops
        testAdj = np.linalg.matrix_power((I_NN + Adj), N)  # check if G is connected

        check = 0
        for i in range(N):
            check += int(bool(len(np.nonzero(Adj[:, i])[0]))) + int(bool(len(np.nonzero(Adj[i])[0])))

        if not np.any(np.any(np.logical_not(testAdj))) and check == 2*N:
            break
    return Adj


def metropolis_hastings(Adj: np.ndarray) -> np.ndarray:
    """Construct a weight matrix using the Metropolis-Hastings method
    
    Args:
        Adj (numpy.ndarray): Adjacency matrix
    
    Returns:
        numpy.ndarray: weighted adjacency matrix
    """
    # weighted adjacency matrix (Metropolis-Hastings)
    N = np.shape(Adj)[0]
    degree = np.sum(Adj, axis=0)
    W = np.zeros([N, N])
    for i in range(N):
        N_i = np.nonzero(Adj[i, :])[0]  # Fixed Neighbors
        for j in N_i:
            W[i, j] = 1/(1+np.max([degree[i], degree[j]]))
        W[i, i] = 1 - np.sum(W[i, :])
    return W


def row_stochastic_matrix(Adj: np.ndarray, weights_type: str='uniform') -> np.ndarray:
    """Construct a row-stochastic weighted adjacency matrix
    
    Args:
        Adj (numpy.ndarray): Adjacency matrix
    
    Returns:
        numpy.ndarray: weighted adjacency matrix
    """
    # row stochastic adjacency matrix 
    N = np.shape(Adj)[0]
    W = np.zeros([N, N])
    if weights_type == 'uniform':
        for i in range(N):
            N_i = len(np.nonzero(Adj[i, :])[0])
            W[i, :] = Adj[i, :]/(N_i + 1)
    elif weights_type == 'random':
        for i in range(N):
            N_i = np.nonzero(Adj[i, :])[0].tolist()
            for j in N_i:
                W[i, j] = np.random.rand()
            self_weight = np.random.rand()
            W[i, :] = W[i, :]/(self_weight + sum(W[i, :])) 
    else:
        raise ValueError('Unknown in_weights type')
    return W


def column_stochastic_matrix(Adj: np.ndarray, weights_type: str='uniform') -> np.ndarray:
    """Construct a column-stochastic weighted adjacency matrix
    
    Args:
        Adj (numpy.ndarray): Adjacency matrix
    
    Returns:
        numpy.ndarray: weighted adjacency matrix
    """
    # column stochastic adjacency matrix 
    W = row_stochastic_matrix(Adj.transpose(), weights_type).transpose()
    return W


class MPIgraph:
    """Create a graph on the network
        
    Args:
        graph_type (str, optional): type of graph ('complete', 'random_binomial'). Defaults to None (complete).
        in_weight_matrix_type (str, optional): type of matrix describing in-neighbors weights ('metropolis', 'row_stochastic', 'column_stochastic'). Defaults to None (metropolis).
        out_weight_matrix_type (str, optional): type of matrix describing out-neighbors weights ('metropolis', 'row_stochastic', 'column_stochastic'). Defaults to None (metropolis).
    """

    def __init__(self, graph_type: str=None, in_weight_matrix_type: str=None, out_weight_matrix_type: str=None, **kwargs):
        if graph_type is None:
            graph_type = 'complete'
        if in_weight_matrix_type is None:
            in_weight_matrix_type = 'metropolis'
        if out_weight_matrix_type is None:
            out_weight_matrix_type = 'metropolis'

        self.number_of_agents = MPI.COMM_WORLD.Get_size()
        self.adjacency_matrix = None
        self.graph_type = None
        self.in_weights = None
        self.out_weights = None

        self.__set_adjacency_matrix(graph_type, **kwargs)
        self.__set_in_weights(in_weight_matrix_type)
        self.__set_out_weights(out_weight_matrix_type)

    def __set_adjacency_matrix(self, graph_type, **kwargs):
        if graph_type == 'complete':
            self.adjacency_matrix = np.ones([self.number_of_agents, self.number_of_agents]) - np.eye(self.number_of_agents)
            self.graph_type = 'undirected'
        elif graph_type == 'random_binomial':
            p = kwargs.get('p', None)
            seed = kwargs.get('seed', None)
            links_type = kwargs.get('links_type', 'undirected')
            self.graph_type = links_type
            self.adjacency_matrix = binomial_random_graph(self.number_of_agents, p, seed, links_type)
        else:
            raise ValueError("Unknown graph type")

    def __set_in_weights(self, weight_matrix_type):
        if weight_matrix_type == 'metropolis':
            self.in_weights = metropolis_hastings(self.adjacency_matrix)
        elif weight_matrix_type == 'row_stochastic':
            self.in_weights = row_stochastic_matrix(self.adjacency_matrix)
        elif weight_matrix_type == 'column_stochastic':
            self.in_weights = column_stochastic_matrix(self.adjacency_matrix)
        else:
            raise ValueError("Unknown in_weights matrix type")
    
    def __set_out_weights(self, weight_matrix_type):
        if weight_matrix_type == 'metropolis':
            self.out_weights = metropolis_hastings(self.adjacency_matrix)
        elif weight_matrix_type == 'row_stochastic':
            self.out_weights = row_stochastic_matrix(self.adjacency_matrix)
        elif weight_matrix_type == 'column_stochastic':
            self.out_weights = column_stochastic_matrix(self.adjacency_matrix)
        else:
            raise ValueError("Unknown in_weights matrix type")

    def get_local_info(self):
        """return the local info available at the agent
        
        Returns:
            tuple: local_rank, in_neighbors, out_neighbors, in_weights, out_weights, 
        """
        local_rank = MPI.COMM_WORLD.Get_rank()
        in_neighbors = np.nonzero(self.adjacency_matrix[local_rank, :])[0].tolist()
        out_neighbors = np.nonzero(self.adjacency_matrix[:, local_rank])[0].tolist()
        in_weights = self.in_weights[local_rank, :].tolist()
        out_weights = self.out_weights[:, local_rank].tolist()
        return local_rank, in_neighbors, out_neighbors, in_weights, out_weights
