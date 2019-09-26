import dill as pickle
import numpy as np
from mpi4py import MPI
from disropt.agents import Agent
from disropt.algorithms.gradient_tracking import GradientTracking
from disropt.functions import QuadraticForm, Variable
from disropt.utils.utilities import is_pos_def
from disropt.utils.graph_constructor import binomial_random_graph, metropolis_hastings
from disropt.problems.problem import Problem

# get MPI info
comm = MPI.COMM_WORLD
nproc = comm.Get_size()
local_rank = comm.Get_rank()

# Generate a common graph (everyone use the same seed)
Adj = binomial_random_graph(nproc, p=0.3, seed=1)
W = metropolis_hastings(Adj)

# reset local seed
np.random.seed()

agent = Agent(
    in_neighbors=np.nonzero(Adj[local_rank, :])[0].tolist(),
    out_neighbors=np.nonzero(Adj[:, local_rank])[0].tolist(),
    in_weights=W[local_rank, :].tolist())

# variable dimension
d = 4

# generate a positive definite matrix
P = np.random.randn(d, d)
while not is_pos_def(P):
    P = np.random.randn(d, d)
bias = np.random.randn(d, 1)
# declare a variable
x = Variable(d)

# define the local objective function
fun = QuadraticForm(x - bias, P)

# local problem
pb = Problem(fun)
agent.set_problem(pb)

# instantiate the algorithms
initial_condition = np.random.rand(d, 1)

algorithm = GradientTracking(agent=agent,
                             initial_condition=initial_condition,
                             enable_log=True)



# run the algorithm
sequence = algorithm.run(iterations=1000, stepsize=0.001)
print("Agent {}: {}".format(agent.id, algorithm.get_result().flatten()))

np.save("agents.npy", nproc)

# save agent and sequence
with open('agent_{}_function.pkl'.format(agent.id), 'wb') as output:
    pickle.dump(agent.problem.objective_function, output, pickle.HIGHEST_PROTOCOL)
np.save("agent_{}_sequence.npy".format(agent.id), sequence)
