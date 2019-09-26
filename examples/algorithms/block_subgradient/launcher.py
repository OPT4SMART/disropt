import dill as pickle
import numpy as np
from mpi4py import MPI
from disropt.agents import Agent
from disropt.algorithms.subgradient import BlockSubgradientMethod
from disropt.functions import QuadraticForm, Variable, SquaredNorm
from disropt.utils.utilities import is_pos_def
from disropt.constraints.projection_sets import Box
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
np.random.seed(local_rank)

agent = Agent(
    in_neighbors=np.nonzero(Adj[local_rank, :])[0].tolist(),
    out_neighbors=np.nonzero(Adj[:, local_rank])[0].tolist(),
    in_weights=W[local_rank, :].tolist())

# variable dimension
n = 6

# generate a positive definite matrix
P = np.random.randn(n, n)
P = P.transpose() @ P
bias = np.random.randn(n, 1)
# declare a variable
x = Variable(n)

# define the local objective function
fn = QuadraticForm(x - bias, P)

# define a (common) constraint set
constr = [x <= 1, x >= -1]

# local problem
pb = Problem(fn, constr)
agent.set_problem(pb)

# instantiate the algorithms
initial_condition = np.random.rand(n, 1)

algorithm = BlockSubgradientMethod(agent=agent,
                                   initial_condition=initial_condition,
                                   enable_log=True)


def step_gen(k):  # define a stepsize generator
    return 0.1/np.sqrt(k+1)


# run the algorithm
sequence = algorithm.run(iterations=10000, stepsize=step_gen, verbose=True)
print("Agent {}: {}".format(agent.id, algorithm.get_result().flatten()))

np.save("agents.npy", nproc)

# save agent and sequence
with open('agent_{}_function.pkl'.format(agent.id), 'wb') as output:
    pickle.dump(agent.problem.objective_function, output, pickle.HIGHEST_PROTOCOL)
np.save("agent_{}_sequence.npy".format(agent.id), sequence)
