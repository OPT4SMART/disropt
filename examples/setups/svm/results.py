import numpy as np
import matplotlib.pyplot as plt
from disropt.problems import Problem
import pickle

# initialize
info  = np.load("info.npy")
NN    = info.N
iters = info.iterations # TODO consider the maximum number of iterations among agents
size  = info.size

# load agent data
sequence = np.zeros((NN, iters, size))
local_constr = {}
for i in range(NN):
    sequence[i,:,:] = np.load("agent_{}_seq.npy".format(i))
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
        cost_err[i, t] = abs(obj_func.eval(sequence[i, t, :]) - cost_centr)
        
# plot cost error
plt.figure()
plt.title('Evolution of cost error')
plt.xlabel(r"iteration $k$")
plt.ylabel(r"$|f(x_i^k) - f^\star|$")

for i in range(NN):
    plt.plot(np.arange(iters), cost_err[i, :])

# plot solution error
sol_err = np.linalg.norm(sequence - x_centr[None, None, :], axis=2)

plt.figure()
plt.title('Evolution of solution error')
plt.xlabel(r"iteration $k$")
plt.ylabel(r"$\|x_i^k - x^\star\|$")

for i in range(NN):
    plt.plot(np.arange(iters), sol_err[i, :])

# TODO plot points and separating hyperplane

plt.show()
