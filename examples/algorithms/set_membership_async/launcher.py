import numpy as np
from mpi4py import MPI
from disropt.agents import Agent
from disropt.algorithms import AsynchronousSetMembership
from disropt.constraints.projection_sets import CircularSector
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


# dimension of the variable
n = 2


def rotate(p, c, theta):  # rotate p around c by theta (rad)
    xr = np.cos(theta)*(p[0]-c[0])-np.sin(theta)*(p[1]-c[1]) + c[0]
    yr = np.sin(theta)*(p[0]-c[0])+np.cos(theta)*(p[1]-c[1]) + c[1]
    return np.array([xr, yr]).reshape(p.shape)


def measure_generator():  # define a measure generator
    np.random.seed(1)
    common_point = np.array([[0.6], [-0.1]])  # np.random.randn(n, 1)
    np.random.seed(10*local_rank)

    position = np.random.randn(n, 1)
    eps_ang = 2*1e-1
    eps_dist = 1e-1

    np.random.seed()
    rd = np.linalg.norm(common_point-position, 2)
    rd2 = rd + eps_dist * np.random.rand()
    measure = position + rd2*(common_point-position)/rd
    rotation = eps_ang * (np.random.rand()-0.5)
    measure = rotate(measure, position, rotation)

    vect = measure - position
    angle = np.arctan2(vect[1], vect[0])
    radius = np.linalg.norm(vect) + eps_dist*np.random.rand()
    return CircularSector(vertex=position,
                          angle=angle,
                          radius=radius,
                          width=2*eps_ang)


# Run algorithm
algorithm = AsynchronousSetMembership(agent,
                                      np.random.rand(n, 1),
                                      enable_log=True,
                                      force_sleep=True,
                                      maximum_sleep=0.01,
                                      sleep_type="random",
                                      force_computation_time=True,
                                      maximum_computation_time=0.1,
                                      computation_time_type="random",
                                      force_unreliable_links=False,
                                      link_failure_probability=0.1)

# assigne measure generator to the agent
algorithm.set_measure_generator(measure_generator)

timestamp_sequence_awake, timestamp_sequence_sleep, sequence = algorithm.run(running_time=10)
print("Agent {}: {}".format(agent.id, algorithm.get_result()))

# Save results
np.save("agents.npy", nproc)
np.save("agent_{}_sequence.npy".format(agent.id), sequence)
np.save("agent_{}_timestamp_sequence_awake.npy".format(agent.id), timestamp_sequence_awake)
np.save("agent_{}_timestamp_sequence_sleep.npy".format(agent.id), timestamp_sequence_sleep)

