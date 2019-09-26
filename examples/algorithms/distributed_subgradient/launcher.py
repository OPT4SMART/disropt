import dill as pickle
import numpy as np
from mpi4py import MPI
from disropt.agents import Agent
from disropt.algorithms.subgradient import SubgradientMethod
from disropt.functions import QuadraticForm, Variable
from disropt.utils.graph_constructor import binomial_random_graph, metropolis_hastings
from disropt.problems import Problem

# get MPI info
comm = MPI.COMM_WORLD
nproc = comm.Get_size()
local_rank = comm.Get_rank()

# Generate a common graph (everyone use the same seed)
Adj = binomial_random_graph(nproc, p=0.3, seed=1)
W = metropolis_hastings(Adj)

# reset local seed
np.random.seed(10*local_rank)

agent = Agent(
    in_neighbors=np.nonzero(Adj[local_rank, :])[0].tolist(),
    out_neighbors=np.nonzero(Adj[:, local_rank])[0].tolist(),
    in_weights=W[local_rank, :].tolist())

# variable dimension
n = 2

# declare a variable
x = Variable(n)

# define the local objective function
P = np.random.rand(n, n)
P = P.transpose() @ P
bias = np.random.rand(n, 1)
fn = QuadraticForm(x - bias, P)


# define a (common) constraint set
constr = [x<= 1, x >= -1]

# local problem
pb = Problem(fn, constr)
agent.set_problem(pb)

# instantiate the algorithms
initial_condition = 10*np.random.rand(n, 1)

algorithm = SubgradientMethod(agent=agent,
                              initial_condition=initial_condition,
                              enable_log=True)


def step_gen(k): # define a stepsize generator
    return 1/((k+1)**0.51)


# run the algorithm
sequence = algorithm.run(iterations=1000, stepsize=step_gen)
print("Agent {}: {}".format(agent.id, algorithm.get_result().flatten()))

np.save("agents.npy", nproc)

# save agent and sequence
np.save("agent_{}_sequence.npy".format(agent.id), sequence)
with open('agent_{}_function.pkl'.format(agent.id), 'wb') as output:
    pickle.dump(fn, output, pickle.HIGHEST_PROTOCOL)
