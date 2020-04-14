import dill as pickle
import numpy as np
from mpi4py import MPI
from disropt.agents import Agent
from disropt.algorithms.constraintexchange import DualDistributedSimplex
from disropt.functions import Variable
from disropt.utils.graph_constructor import ring_graph
from disropt.utils.LP_utils import generate_LP
from disropt.problems import LinearProblem

# get MPI info
comm = MPI.COMM_WORLD
nproc = comm.Get_size()
local_rank = comm.Get_rank()

# Generate a ring graph (for which the diameter is nproc-1)
Adj = ring_graph(nproc)
graph_diam = nproc-1

# reset local seed
np.random.seed(1)

# number of variables
n_vars = 3

# number of constraints for each processor
k = 2

# generate a feasible optimization problem with k * nproc constraints
c_glob, A_glob, b_glob, x_glob = generate_LP(n_vars, k * nproc, 50, direction='max')

# extract the constraints assigned to this agent
local_indices = list(np.arange(k*local_rank, k*(local_rank+1)))

c_loc = c_glob
A_loc = A_glob[local_indices, :]
b_loc = b_glob[local_indices, :]

# define the local problem data
x = Variable(n_vars)
obj = -c_loc @ x # minus sign for maximization
constr = A_loc.transpose() @ x <= b_loc
problem = LinearProblem(objective_function=obj, constraints=constr)

# create agent
agent = Agent(in_neighbors=np.nonzero(Adj[local_rank, :])[0].tolist(),
        out_neighbors=np.nonzero(Adj[:, local_rank])[0].tolist())
agent.problem = problem

# instantiate the algorithm
algorithm = DualDistributedSimplex(agent, enable_log=True, num_constraints=nproc*k,
    local_indices=local_indices, stop_iterations=2*graph_diam+1)

# run the algorithm
x_sequence, J_sequence = algorithm.run(iterations=100, verbose=True)

# print results
_, _, _, J_final = algorithm.get_result()
print("Agent {} - {} iterations - final cost {}".format(agent.id, len(J_sequence), J_final))

# save results to file
if agent.id == 0:
    with open('info.pkl', 'wb') as output:
        pickle.dump({'N': nproc, 'n_constr': k * nproc, 'n_vars': n_vars, 'c': c_glob,
            'A': A_glob, 'b': b_glob, 'opt_sol': x_glob}, output, pickle.HIGHEST_PROTOCOL)

np.save("agent_{}_x_seq.npy".format(agent.id), x_sequence)
np.save("agent_{}_J_seq.npy".format(agent.id), J_sequence)
