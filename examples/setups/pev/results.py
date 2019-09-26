import numpy as np
import matplotlib.pyplot as plt
from disropt.problems import Problem
import pickle

# initialize
with open('info.pkl', 'rb') as inp:
    info = pickle.load(inp)
NN    = info['N']
iters = info['iterations']
SS    = info['n_coupling']

# load agent data
seq_dds = {}
seq_dpd = {}
local_obj_func = {}
local_coup_func = {}
for i in range(NN):
    seq_dds[i] = np.load("agent_{}_seq_dds.npy".format(i))
    seq_dpd[i] = np.load("agent_{}_seq_dpd.npy".format(i))
    with open('agent_{}_objective_func.pkl'.format(i), 'rb') as inp:
        local_obj_func[i] = pickle.load(inp)
    with open('agent_{}_coupling_func.pkl'.format(i), 'rb') as inp:
        local_coup_func[i] = pickle.load(inp)

# compute costs and coupling values
cost_seq_dds = np.zeros(iters)
cost_seq_dpd = np.zeros(iters)
coup_seq_dds = np.zeros((iters, SS))
coup_seq_dpd = np.zeros((iters, SS))

for t in range(iters):
    for i in range(NN):
        cost_seq_dds[t] += local_obj_func[i].eval(seq_dds[i][t, :])
        cost_seq_dpd[t] += local_obj_func[i].eval(seq_dpd[i][t, :])
        coup_seq_dds[t] += np.squeeze(local_coup_func[i].eval(seq_dds[i][t, :]))
        coup_seq_dpd[t] += np.squeeze(local_coup_func[i].eval(seq_dpd[i][t, :]))

# plot cost
plt.figure()
plt.title('Evolution of cost')
plt.xlabel(r"iteration $k$")
plt.ylabel(r"$\sum_{i=1}^N f_i(x_i^k)$")
plt.plot(np.arange(iters), cost_seq_dds, label='Distributed Dual Subgradient')
plt.plot(np.arange(iters), cost_seq_dpd, label='Distributed Primal Decomposition')
plt.legend()

# plot maximum coupling constraint value
plt.figure()
plt.title('Evolution of maximum coupling constraint value')
plt.xlabel(r"iteration $k$")
plt.ylabel(r"$\max_{s} \: \sum_{i=1}^N g_{is}(x_i^k)$")
plt.plot(np.arange(iters), np.amax(coup_seq_dds, axis=1), label='Distributed Dual Subgradient')
plt.plot(np.arange(iters), np.amax(coup_seq_dpd, axis=1), label='Distributed Primal Decomposition')
plt.legend()

plt.show()