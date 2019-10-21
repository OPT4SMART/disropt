import dill as pickle
import numpy as np
import matplotlib.pyplot as plt
from disropt.problems import Problem

N = np.load("agents.npy")
n = 2

lambda_sequence = {}
x_sequence = {}
z_sequence = {}
local_obj_function = {}
for i in range(N):
    with open('agent_{}_dual_sequence.pkl'.format(i), 'rb') as input:
        lambda_sequence[i] = pickle.load(input)
    x_sequence[i] = np.load("agent_{}_primal_sequence.npy".format(i))
    z_sequence[i] = np.load("agent_{}_auxiliary_primal_sequence.npy".format(i))
    with open('agent_{}_function.pkl'.format(i), 'rb') as input:
        local_obj_function[i] = pickle.load(input)
with open('constraints.pkl', 'rb') as input:
    constr = pickle.load(input)
iters = x_sequence[0].shape[0]

# solve centralized problem
global_obj_func = 0
for i in range(N):
    global_obj_func += local_obj_function[i]

global_pb = Problem(global_obj_func, constr)
x_centr = global_pb.solve()
cost_centr = global_obj_func.eval(x_centr)
x_centr = x_centr.flatten()

# compute cost errors
cost_err = np.zeros((N, iters)) - cost_centr

for i in range(N):
    for t in range(iters):
        # add i-th cost
        cost_err[i, t] += local_obj_function[i].eval(x_sequence[i][t, :])

# plot maximum cost error
plt.figure()
plt.title('Maximum cost error (among agents)')
plt.xlabel(r"iteration $k$")
plt.ylabel(r"$\max_{i} \: \left|\sum_{j=1}^N f_j(x_i^k) - f^\star \right|$")
plt.semilogy(np.arange(iters), np.amax(np.abs(cost_err), axis=0))

# plot maximum solution error
sol_err = np.zeros((N, iters))
for i in range(N):
    sol_err[i] = np.linalg.norm(np.squeeze(x_sequence[i]) - x_centr[None, :], axis=1)

plt.figure()
plt.title('Maximum solution error (among agents)')
plt.xlabel(r"iteration $k$")
plt.ylabel(r"$\max_{i} \: \|x_i^k - x^\star\|$")
plt.semilogy(np.arange(iters), np.amax(sol_err, axis=0))

plt.show()
