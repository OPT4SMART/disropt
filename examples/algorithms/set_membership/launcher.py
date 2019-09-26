import numpy as np
from mpi4py import MPI
from disropt.agents import Agent
from disropt.algorithms import SetMembership
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
np.random.seed(10*local_rank)


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
    common_point = np.random.randn(n, 1)
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
algorithm = SetMembership(agent, np.random.rand(n, 1), enable_log=True)
# assign measure generator to the agent
algorithm.set_measure_generator(measure_generator)
sequence = algorithm.run(iterations=1000)

# print solution
print("Agent {}: {}".format(agent.id, algorithm.get_result()))

# save data
np.save("agents.npy", nproc)
np.save("agent_{}_sequence.npy".format(agent.id), sequence)
