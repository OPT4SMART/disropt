###############################################################
# CONSTRAINT-COUPLED Example
# Charging of Plug-in Electric Vehicles (PEVs)
#
# The problem consists of finding an optimal overnight schedule to
# charge electric vehicles. See [Vu16] for the problem model
# and generation parameters.
#
# Note: in this example we consider the "charging-only" case
###############################################################
# Compared Algorithms:
#
# - Distributed Dual Subgradient
# - Distributed Primal Decomposition
###############################################################

import dill as pickle
import numpy as np
from numpy.random import uniform as rnd
from mpi4py import MPI
from disropt.agents import Agent
from disropt.algorithms import DualSubgradientMethod, PrimalDecomposition
from disropt.functions import Variable
from disropt.utils.graph_constructor import binomial_random_graph, metropolis_hastings, binomial_random_graph_sequence
from disropt.problems import ConstraintCoupledProblem

# get MPI info
NN = MPI.COMM_WORLD.Get_size()
agent_id = MPI.COMM_WORLD.Get_rank()

# Generate a common graph (everyone uses the same seed)
Adj = binomial_random_graph(NN, p=0.2, seed=1)
W = metropolis_hastings(Adj)

# generate edge probabilities
edge_prob = np.random.uniform(0.3, 0.9, (NN, NN))
edge_prob[Adj == 0] = 0
i_lower = np.tril_indices(NN)
edge_prob[i_lower] = edge_prob.T[i_lower] # symmetrize

#####################
# Generate problem parameters
#####################

# Problem parameters are generated according to the table in [Vu16]

#### Common parameters

TT = 24 # number of time windows
DeltaT = 20 # minutes
# PP_max = 3 * NN # kWh
PP_max = 0.5 * NN # kWh
CC_u = rnd(19,35, (TT, 1)) # EUR/MWh - TT entries

#### Individual parameters

np.random.seed(10*agent_id)

PP = rnd(3,5) # kW
EE_min = 1 # kWh
EE_max = rnd(8,16) # kWh
EE_init = rnd(0.2,0.5) * EE_max # kWh
EE_ref = rnd(0.55,0.8) * EE_max # kWh
zeta_u = 1 - rnd(0.015, 0.075) # pure number

#####################
# Generate problem object
#####################

# normalize unit measures
DeltaT = DeltaT/60 # minutes  -> hours
CC_u = CC_u/1e3    # Euro/MWh -> Euro/KWh

# optimization variables
z = Variable(2*TT + 1) # stack of e (state of charge) and u (input charging power)
e = np.vstack((np.eye(TT+1), np.zeros((TT, TT+1)))) @ z # T+1 components (from 0 to T)
u = np.vstack((np.zeros((TT+1, TT)), np.eye(TT))) @ z   # T components (from 0 to T-1)

# objective function
obj_func = PP * (CC_u @ u)

# coupling function
coupling_func = PP*u - (PP_max/NN)

# local constraints
e_0 = np.zeros((TT+1, 1))
e_T = np.zeros((TT+1, 1))
e_0[0] = 1
e_T[TT] = 1
constr = [e_0 @ e == EE_init, e_T @ e >= EE_ref] # feedback and reference constraints

for kk in range(0, TT):
    e_cur = np.zeros((TT+1, 1))
    u_cur = np.zeros((TT, 1))
    e_new = np.zeros((TT+1, 1))
    e_cur[kk] = 1
    u_cur[kk] = 1
    e_new[kk+1] = 1
    constr.append(e_new @ e == e_cur @ e + PP*DeltaT*zeta_u*u_cur @ u) # dynamics
    constr.extend([u_cur @ u <= 1, u_cur @ u >= 0]) # input constraints
    constr.extend([e_new @ e <= EE_max, e_new @ e >= EE_min]) # state constraints

#####################
# Distributed algorithms
#####################

# local agent and problem
agent = Agent(
    in_neighbors=np.nonzero(Adj[agent_id, :])[0].tolist(),
    out_neighbors=np.nonzero(Adj[:, agent_id])[0].tolist(),
    in_weights=W[agent_id, :].tolist())

pb = ConstraintCoupledProblem(obj_func, constr, coupling_func)
agent.set_problem(pb)

# instantiate the algorithms
# y0  = np.zeros((TT, 1))
# mu0 = np.zeros((TT, 1))
y0 = 10*np.random.rand(TT, 1)
mu0 = 10*np.random.rand(TT, 1)

theothers = [i for i in range(NN) if i != agent_id]
y_others = agent.communicator.neighbors_exchange(y0, theothers, theothers, False)
y_others[agent_id] = y0
y_mean = sum([x for _, x in y_others.items()])/NN
y0 -= y_mean

dds = DualSubgradientMethod(agent=agent,
                            initial_condition=mu0,
                            enable_log=True)

dpd = PrimalDecomposition  (agent=agent,
                            initial_condition=y0,
                            enable_log=True)

############################################

num_iterations = 1000

# generate sequence of adjacency matrices
Adj_seq = binomial_random_graph_sequence(Adj, num_iterations, edge_prob, NN, 1)

def step_gen(k): # define a stepsize generator
    return 1/((k+1)**0.6)
# def step_gen(k): # define a stepsize generator
#     return 0.01

def update_graph(k):
    Adj_k = Adj_seq[k]
    W_k = metropolis_hastings(Adj_k)
    agent.set_neighbors(in_neighbors=np.nonzero(Adj_k[agent_id, :])[0].tolist(),
                        out_neighbors=np.nonzero(Adj_k[:, agent_id])[0].tolist())
    agent.set_weights(in_weights=W_k[agent_id, :].tolist())

# run the algorithms
if agent_id == 0:
    print("Distributed dual subgradient")
_, dds_seq = dds.run(iterations=num_iterations, stepsize=step_gen, verbose=True)

if agent_id == 0:
    print("Distributed primal decomposition")
dpd_seq, _, _ = dpd.run(iterations=num_iterations, stepsize=step_gen, M=30.0, verbose=True)

# save information
if agent_id == 0:
    with open('info.pkl', 'wb') as output:
        pickle.dump({'N': NN, 'iterations': num_iterations, 'n_coupling': TT}, output, pickle.HIGHEST_PROTOCOL)

with open('agent_{}_objective_func.pkl'.format(agent_id), 'wb') as output:
    pickle.dump(obj_func, output, pickle.HIGHEST_PROTOCOL)
with open('agent_{}_coupling_func.pkl'.format(agent_id), 'wb') as output:
    pickle.dump(coupling_func, output, pickle.HIGHEST_PROTOCOL)
with open('agent_{}_local_constr.pkl'.format(agent_id), 'wb') as output:
    pickle.dump(constr, output, pickle.HIGHEST_PROTOCOL)
np.save("agent_{}_seq_dds.npy".format(agent_id), dds_seq)
np.save("agent_{}_seq_dpd.npy".format(agent_id), dpd_seq)