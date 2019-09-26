import dill as pickle
import numpy as np
from mpi4py import MPI
from disropt.agents import Agent
from disropt.algorithms.dual_decomp import DualDecomposition
from disropt.functions import QuadraticForm, Variable
from disropt.utils.graph_constructor import binomial_random_graph
from disropt.problems import Problem

# get MPI info
comm = MPI.COMM_WORLD
nproc = comm.Get_size()
local_rank = comm.Get_rank()

# Generate a common graph (everyone use the same seed)
Adj = binomial_random_graph(nproc, p=0.3, seed=1)

# reset local seed
np.random.seed(10*local_rank)

agent = Agent(
    in_neighbors=np.nonzero(Adj[local_rank, :])[0].tolist(),
    out_neighbors=np.nonzero(Adj[:, local_rank])[0].tolist())

# variable dimension
n = 2

# generate a positive definite matrix
P = np.random.randn(n, n)
P = P.transpose() @ P
bias = 3*np.random.randn(n, 1)
# declare a variable
x = Variable(n)

# define the local objective function
fn = QuadraticForm(x - bias, P)

# define a (common) constraint set
constr = [np.eye(n) @ x <= 1, np.eye(n) @ x >= -1]

# local problem
pb = Problem(fn, constr)
agent.set_problem(pb)

# instantiate the algorithms
initial_condition = {}

for j in agent.in_neighbors:
    initial_condition[j] = 10*np.random.rand(n, 1)

algorithm = DualDecomposition(agent=agent,
                              initial_condition=initial_condition,
                              enable_log=True)


def step_gen(k): # define a stepsize generator
    return 1/((k+1)**0.51)


# run the algorithm
x_sequence, lambda_sequence = algorithm.run(iterations=100, stepsize=step_gen)
x_t, lambda_t = algorithm.get_result()
print("Agent {}: primal {} dual {}".format(agent.id, x_t.flatten(), lambda_t))

np.save("agents.npy", nproc)

# save agent and sequence
with open('agent_{}_function.pkl'.format(agent.id), 'wb') as output:
    pickle.dump(agent.problem.objective_function, output, pickle.HIGHEST_PROTOCOL)
with open('agent_{}_dual_sequence.pkl'.format(agent.id), 'wb') as output:
    pickle.dump(lambda_sequence, output, pickle.HIGHEST_PROTOCOL)
np.save("agent_{}_primal_sequence.npy".format(agent.id), x_sequence)
