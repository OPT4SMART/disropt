import numpy as np
from mpi4py import MPI
from disropt.agents import Agent
from disropt.algorithms.consensus import BlockConsensus
from disropt.utils.graph_constructor import binomial_random_graph, metropolis_hastings

# get MPI info
comm = MPI.COMM_WORLD
nproc = comm.Get_size()
local_rank = comm.Get_rank()

# Generate a common graph (everyone use the same seed)
Adj = binomial_random_graph(nproc, p=0.3, seed=1)
W = metropolis_hastings(Adj)

# reset local seed
np.random.seed()

# create local agent
agent = Agent(in_neighbors=np.nonzero(Adj[local_rank, :])[0].tolist(),
              out_neighbors=np.nonzero(Adj[:, local_rank])[0].tolist(),
              in_weights=W[local_rank, :].tolist())

# instantiate the consensus algorithm
n = 4  # decision variable dimension (n, 1)
x0 = np.random.rand(n, 1)
algorithm = BlockConsensus(agent=agent,
                           initial_condition=x0,
                           enable_log=True,
                           blocks_list=[(0, 1), (2, 3)],
                           probabilities=[0.3, 0.7])
# run the algorithm
sequence = algorithm.run(iterations=100)

# print solution
print("Agent {}: {}".format(agent.id, algorithm.get_result()))

# save data
np.save("agents.npy", nproc)
np.save("agent_{}_sequence.npy".format(agent.id), sequence)
