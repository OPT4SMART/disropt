import dill as pickle
import numpy as np
from mpi4py import MPI
from disropt.agents import Agent
from disropt.algorithms.primal_decomp import PrimalDecomposition
from disropt.functions import QuadraticForm, Variable
from disropt.utils.utilities import is_pos_def
from disropt.constraints.projection_sets import Box
from disropt.utils.graph_constructor import binomial_random_graph
from disropt.problems.constraint_coupled_problem import ConstraintCoupledProblem

# get MPI info
comm = MPI.COMM_WORLD
nproc = comm.Get_size()
local_rank = comm.Get_rank()

# Generate a common graph (everyone use the same seed)
Adj = binomial_random_graph(nproc, p=0.3, seed=1)

# reset local seed
np.random.seed()

agent = Agent(
    in_neighbors=np.nonzero(Adj[local_rank, :])[0].tolist(),
    out_neighbors=np.nonzero(Adj[:, local_rank])[0].tolist())

# local variable dimension - random in [2,5]
n_i = np.random.randint(2, 6)

# number of coupling constraints
S = 3

# generate a positive definite matrix
P = np.random.randn(n_i, n_i)
P = P.transpose() @ P
bias = np.random.randn(n_i, 1)

# declare a variable
x = Variable(n_i)

# define the local objective function
fn = QuadraticForm(x - bias, P)

# define the local constraint set
low = -2*np.ones((n_i, 1))
up = 2*np.ones((n_i, 1))
constr = Box(low, up)

# define the local contribution to the coupling constraints
A = np.random.randn(S, n_i)
coupling_fn = A.transpose() @ x

# create local problem and assign to agent
pb = ConstraintCoupledProblem(objective_function=fn,
                              constraints=constr,
                              coupling_function=coupling_fn)
agent.set_problem(pb)

# initialize allocation
y0 = np.zeros((S, 1))

algorithm = PrimalDecomposition(agent=agent,
                                initial_condition=y0,
                                enable_log=True)


def step_gen(k): # define a stepsize generator
    return 0.1/np.sqrt(k+1)

# run the algorithm
x_sequence, y_sequence = algorithm.run(iterations=1000, stepsize=step_gen, M=100.0)
x_t, y_t = algorithm.get_result()
print("Agent {}: primal {} allocation {}".format(agent.id, x_t.flatten(), y_t.flatten()))

np.save("agents.npy", nproc)

# save agent and sequence
with open('agent_{}_obj_function.pkl'.format(agent.id), 'wb') as output:
    pickle.dump(agent.problem.objective_function, output, pickle.HIGHEST_PROTOCOL)
with open('agent_{}_coup_function.pkl'.format(agent.id), 'wb') as output:
    pickle.dump(agent.problem.coupling_function, output, pickle.HIGHEST_PROTOCOL)
np.save("agent_{}_allocation_sequence.npy".format(agent.id), y_sequence)
np.save("agent_{}_primal_sequence.npy".format(agent.id), x_sequence)
