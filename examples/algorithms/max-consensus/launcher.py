import numpy as np
from mpi4py import MPI
from disropt.agents import Agent
from disropt.algorithms.misc import MaxConsensus
from disropt.utils.graph_constructor import ring_graph

# get MPI info
comm = MPI.COMM_WORLD
nproc = comm.Get_size()
local_rank = comm.Get_rank()

# Generate a common graph (everyone use the same seed)
Adj = ring_graph(nproc)
graph_diam = nproc-1
n_vars = 3

# create local agent
agent = Agent(in_neighbors=np.nonzero(Adj[local_rank, :])[0].tolist(),
              out_neighbors=np.nonzero(Adj[:, local_rank])[0].tolist())

# instantiate the max-consensus algorithm
np.random.seed(local_rank*5)

x0 = np.random.rand(n_vars)*10
algorithm = MaxConsensus(agent, x0, graph_diam, True)

print('Initial value of agent {}:\n {}'.format(local_rank, x0))

# execute algorithm
x_sequence = algorithm.run(iterations=100)

# get result
x_final = algorithm.get_result()

print('Result of agent {}:\n {}'.format(local_rank, x_final))