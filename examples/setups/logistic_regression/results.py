import numpy as np
import matplotlib.pyplot as plt
from disropt.problems import Problem
import pickle

# initialize
with open('info.pkl', 'rb') as inp:
    info = pickle.load(inp)
NN    = info['N']
iters = info['iterations']
size  = info['size']

# load agent data
seq_subgr   = np.zeros((NN, iters, size))
seq_gradtr  = np.zeros((NN, iters, size))
local_function = {}
for i in range(NN):
    seq_subgr[i,:,:]   = np.load("agent_{}_seq_subgr.npy".format(i))
    seq_gradtr[i,:,:]  = np.load("agent_{}_seq_gradtr.npy".format(i))
    with open('agent_{}_func.pkl'.format(i), 'rb') as inp:
        local_function[i] = pickle.load(inp)

# solve centralized problem
global_obj_func = 0
for i in range(NN):
    global_obj_func += local_function[i]

global_pb = Problem(global_obj_func)
x_centr = global_pb.solve()
cost_centr = global_obj_func.eval(x_centr)
x_centr = x_centr.flatten()

# compute cost errors
cost_err_subgr   = np.zeros((NN, iters))
cost_err_gradtr  = np.zeros((NN, iters))

for i in range(NN):
    for t in range(iters):
        # first compute global function value at local point
        cost_ii_tt_subgr   = 0
        cost_ii_tt_gradtr  = 0
        for j in range(NN):
            cost_ii_tt_subgr   += local_function[j].eval(seq_subgr[i, t, :][:, None])
            cost_ii_tt_gradtr  += local_function[j].eval(seq_gradtr[i, t, :][:, None])
        
        # then compute errors
        cost_err_subgr[i, t]   = abs(cost_ii_tt_subgr - cost_centr)
        cost_err_gradtr[i, t]  = abs(cost_ii_tt_gradtr - cost_centr)

# plot maximum cost error
plt.figure()
plt.title('Maximum cost error (among agents)')
plt.xlabel(r"iteration $k$")
plt.ylabel(r"$\max_{i} \: \left|\sum_{j=1}^N f_j(x_i^k) - f^\star \right|$")
plt.semilogy(np.arange(iters), np.amax(cost_err_subgr, axis=0), label='Distributed Subgradient')
plt.semilogy(np.arange(iters), np.amax(cost_err_gradtr, axis=0), label='Gradient Tracking')
plt.legend()

# plot maximum solution error
sol_err_subgr   = np.linalg.norm(seq_subgr - x_centr[None, None, :], axis=2)
sol_err_gradtr  = np.linalg.norm(seq_gradtr - x_centr[None, None, :], axis=2)

plt.figure()
plt.title('Maximum solution error (among agents)')
plt.xlabel(r"iteration $k$")
plt.ylabel(r"$\max_{i} \: \|x_i^k - x^\star\|$")
plt.semilogy(np.arange(iters), np.amax(sol_err_subgr, axis=0), label='Distributed Subgradient')
plt.semilogy(np.arange(iters), np.amax(sol_err_gradtr, axis=0), label='Gradient Tracking')
plt.legend()

plt.show()