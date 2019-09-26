import numpy as np
from mpi4py import MPI
from disropt.agents import Agent
from disropt.algorithms import AsynchronousConsensus
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
agent = Agent(
    in_neighbors=np.nonzero(Adj[local_rank, :])[0].tolist(),
    out_neighbors=np.nonzero(Adj[:, local_rank])[0].tolist(),
    in_weights=W[local_rank, :].tolist())

# instantiate the asynchronous consensus algorithm
x0 = np.random.rand(2, 1)
algorithm = AsynchronousConsensus(agent=agent,
                                  initial_condition=x0,
                                  enable_log=True,
                                  force_sleep=True,
                                  maximum_sleep=0.1,
                                  sleep_type="random",
                                  force_computation_time=True,
                                  maximum_computation_time=0.1,
                                  computation_time_type="random",
                                  force_unreliable_links=True,
                                  link_failure_probability=0.1)

# run the algorithm
timestamp_sequence_awake, timestamp_sequence_sleep, sequence = algorithm.run(running_time=4)

# print solution
print("Agent {}: {}".format(agent.id, algorithm.get_result()))

# save data
np.save("agents.npy", nproc)
np.save("agent_{}_sequence.npy".format(agent.id), sequence)
np.save("agent_{}_timestamp_sequence_awake.npy".format(agent.id), timestamp_sequence_awake)
np.save("agent_{}_timestamp_sequence_sleep.npy".format(agent.id), timestamp_sequence_sleep)
