import numpy as np
import matplotlib.pyplot as plt
from disropt.problems import Problem
import pickle
import tikzplotlib

# initialize
with open('info.pkl', 'rb') as inp:
    info = pickle.load(inp)
NN = info['N']
iters = info['iterations']
size = info['size']

# load agent data
sequence = np.zeros((NN, iters, size))
local_constr = {}
for i in range(NN):
    sequence[i, :, :] = np.load("agent_{}_seq.npy".format(i), allow_pickle=True).reshape((iters, size))
    with open('agent_{}_constr.pkl'.format(i), 'rb') as inp:
        local_constr[i] = pickle.load(inp)
with open('objective_function.pkl', 'rb') as inp:
    obj_func = pickle.load(inp)

# solve centralized problem
global_constr = []
for i in range(NN):
    global_constr.extend(local_constr[i])
global_pb = Problem(obj_func, global_constr)
x_centr = global_pb.solve()
cost_centr = obj_func.eval(x_centr)

# compute cost errors
cost_err = np.zeros((NN, iters))

for i in range(NN):
    for t in range(iters):
        cost_err[i, t] = abs(obj_func.eval(sequence[i, t, :].reshape((size, 1))) - cost_centr)

# compute max violation
vio_err = np.zeros((NN, iters))
for i in range(NN):
    for t in range(iters):
        xt = sequence[i, t, :].reshape((size, 1))
        max_err = np.zeros((len(global_constr), 1))
        for c in range(len(global_constr)):
            max_err[c] = global_constr[c].function.eval(xt)
        vio_err[i, t] = np.max(max_err)

# Plot the evolution of the local estimates
# generate N colors
colors = {}
for i in range(NN):
    colors[i] = np.random.rand(3, 1).flatten()

# plot cost error
plt.figure()
plt.title('Evolution of cost error')
plt.xlabel(r"iteration $k$")
plt.ylabel(r"$|f(x_i^k) - f^\star|$")

for i in range(NN):
    plt.plot(np.arange(iters), cost_err[i, :], color=colors[i])

# # plot violation error
plt.figure()
plt.title('Maximum constraint violation')
plt.xlabel(r"iteration $k$")
plt.ylabel(r"$max_j g(x_i^k)$")

for i in range(NN):
    plt.plot(np.arange(iters), vio_err[i, :], color=colors[i])

plt.show()
