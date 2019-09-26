import numpy as np
import networkx as nx
from mpi4py import MPI
from disropt.agents import Agent
from disropt.algorithms.misc import LogicAnd
from disropt.utils.graph_constructor import binomial_random_graph, metropolis_hastings

# get MPI info
comm = MPI.COMM_WORLD
nproc = comm.Get_size()
local_rank = comm.Get_rank()

# Generate a common graph (everyone use the same seed)
Adj = binomial_random_graph(nproc, p=0.1, seed=1)
W = metropolis_hastings(Adj)
graph = nx.DiGraph(Adj)
graph_diameter = nx.diameter(graph)

# create local agent
agent = Agent(in_neighbors=np.nonzero(Adj[local_rank, :])[0].tolist(),
              out_neighbors=np.nonzero(Adj[:, local_rank])[0].tolist(),
              in_weights=W[local_rank, :].tolist())

# instantiate the logic-and algorithm
flag = True  
algorithm = LogicAnd(agent, graph_diameter, flag=flag)

algorithm.run(maximum_iterations=100)

print(algorithm.S)
