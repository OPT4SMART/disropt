import numpy as np
import matplotlib.pyplot as plt
from disropt.problems import Problem
from disropt.functions import ExtendedFunction
from disropt.constraints import ExtendedConstraint
import dill as pickle

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
local_constr = {}
for i in range(NN):
    seq_dds[i] = np.load("agent_{}_seq_dds.npy".format(i))
    seq_dpd[i] = np.load("agent_{}_seq_dpd.npy".format(i))
    with open('agent_{}_objective_func.pkl'.format(i), 'rb') as inp:
        local_obj_func[i] = pickle.load(inp)
    with open('agent_{}_coupling_func.pkl'.format(i), 'rb') as inp:
        local_coup_func[i] = pickle.load(inp)
    with open('agent_{}_local_constr.pkl'.format(i), 'rb') as inp:
        local_constr[i] = pickle.load(inp)

# solve centralized problem
dim = sum([f.input_shape[0] for _, f in local_obj_func.items()]) # size of overall variable
centr_obj_func = 0
centr_constr = []
centr_coupling_func = 0
pos = 0
for i in range(NN):
    n_i = local_obj_func[i].input_shape[0]

    centr_obj_func += ExtendedFunction(local_obj_func[i], n_var = dim-n_i, pos=pos)
    centr_constr.extend(ExtendedConstraint(local_constr[i], n_var = dim-n_i, pos=pos))
    centr_coupling_func += ExtendedFunction(local_coup_func[i], n_var = dim-n_i, pos=pos)

    pos += n_i

centr_constr.append(centr_coupling_func <= 0)
global_pb = Problem(centr_obj_func, centr_constr)
x_centr = global_pb.solve()
cost_centr = centr_obj_func.eval(x_centr).flatten()
x_centr = x_centr.flatten()

# compute costs and coupling values
cost_seq_dds = np.zeros(iters) - cost_centr
cost_seq_dpd = np.zeros(iters) - cost_centr
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
plt.title('Evolution of cost error')
plt.xlabel(r"iteration $k$")
plt.ylabel(r"$| (\sum_{i=1}^N f_i(x_i^k) - f^\star)/f^\star |$")
plt.semilogy(np.arange(iters), np.abs(cost_seq_dds/cost_centr), label='Distributed Dual Subgradient')
plt.semilogy(np.arange(iters), np.abs(cost_seq_dpd/cost_centr), label='Distributed Primal Decomposition')
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