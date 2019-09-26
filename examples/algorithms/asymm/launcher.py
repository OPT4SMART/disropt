import numpy as np
import networkx as nx
from mpi4py import MPI
from disropt.agents import Agent
from disropt.algorithms.asymm import ASYMM
from disropt.utils.graph_constructor import binomial_random_graph, metropolis_hastings
from disropt.functions import SquaredNorm, Norm, Variable
from disropt.problems import Problem

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


# problem set-up
n = 2

# target point
x_true = np.random.randn(n, 1)

# reset local seed
np.random.seed(local_rank)

# local position
c = np.random.randn(n, 1)

# true distance
distance = np.linalg.norm(x_true-c, ord=2)

# declare a variable
x = Variable(n)

# define the local objective function
objective = x@x

# local constraint
upper_bound = (x-c)@(x-c) <= (distance**2 + 0.0001*np.random.rand())
lower_bound = (x-c)@(x-c) >= (distance**2 - 0.0001*np.random.rand())
constr = SquaredNorm(x-c) == distance**2

constraints = [upper_bound, lower_bound]
# define local problem
pb = Problem(objective_function=objective, constraints=constraints)
agent.set_problem(pb)

####
x0 = np.random.randn(n, 1)
algorithm = ASYMM(agent=agent,
                  graph_diameter=graph_diameter,
                  initial_condition=x0,
                  enable_log=True)

timestamp_sequence_awake, timestamp_sequence_sleep, sequence = algorithm.run(running_time=5.0)

# print solution
print(x_true)
print("Agent {}: {}".format(agent.id, algorithm.get_result()))

# save data
np.save("agents.npy", nproc)
np.save("agent_{}_sequence.npy".format(agent.id), sequence)
np.save("agent_{}_timestamp_sequence_awake.npy".format(agent.id), timestamp_sequence_awake)
np.save("agent_{}_timestamp_sequence_sleep.npy".format(agent.id), timestamp_sequence_sleep)
