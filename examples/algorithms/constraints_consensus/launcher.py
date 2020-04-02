import dill as pickle
import numpy as np
from mpi4py import MPI
from disropt.agents import Agent
from disropt.algorithms import ConstraintsConsensus
from disropt.functions import Variable, QuadraticForm
from disropt.utils.graph_constructor import binomial_random_graph
from disropt.problems import Problem

# get MPI info
NN = MPI.COMM_WORLD.Get_size()
agent_id = MPI.COMM_WORLD.Get_rank()

# Generate a common graph (everyone uses the same seed)
Adj = binomial_random_graph(NN, p=0.03, seed=1)

#####################
# Problem parameters
#####################
np.random.seed(10*agent_id)

# linear objective function
dim = 2
z = Variable(dim)
c = np.ones([dim,1])
obj_func = c @ z

# constraints are circles of the form (z-p)^\top (z-p) <= 1
# equivalently z^\top z - 2(A p)^\top z + p^\top A p <= 1
I = np.eye(dim)
p = np.random.rand(dim,1)
r = 1 # unitary radius

constr = []
ff = QuadraticForm(z,I,- 2*(I @ p),(p.transpose() @ I @ p) - r**2)
constr.append(ff<= 0)

#####################
# Distributed algorithms
#####################

# local agent and problem
agent = Agent(
    in_neighbors=np.nonzero(Adj[agent_id, :])[0].tolist(),
    out_neighbors=np.nonzero(Adj[:, agent_id])[0].tolist())
pb = Problem(obj_func, constr)
agent.set_problem(pb)

# instantiate the algorithm
algorithm = ConstraintsConsensus(agent=agent,
                                 enable_log=True)

# run the algorithm
n_iter = NN*3
x_sequence = algorithm.run(iterations=n_iter, verbose=True)

# print results
x_final = algorithm.get_result()
print("Agent {}: {}".format(agent_id, x_final.flatten()))

# save results to file
if agent_id == 0:
    with open('info.pkl', 'wb') as output:
        pickle.dump({'N': NN, 'size': dim, 'iterations': n_iter}, output, pickle.HIGHEST_PROTOCOL)
    with open('objective_function.pkl', 'wb') as output:
        pickle.dump(obj_func, output, pickle.HIGHEST_PROTOCOL)

with open('agent_{}_constr.pkl'.format(agent_id), 'wb') as output:
    pickle.dump(constr, output, pickle.HIGHEST_PROTOCOL)
np.save("agent_{}_seq.npy".format(agent_id), x_sequence)
