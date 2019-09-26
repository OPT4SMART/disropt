import numpy as np
import matplotlib.pyplot as plt
import pickle

N = np.load("agents.npy")
S = 3

lambda_sequence = {}
xhat_sequence = {}
local_obj_function = {}
local_coup_function = {}
for i in range(N):
    lambda_sequence[i] = np.load("agent_{}_dual_sequence.npy".format(i))
    xhat_sequence[i] = np.load("agent_{}_runavg_sequence.npy".format(i))
    with open('agent_{}_obj_function.pkl'.format(i), 'rb') as input:
        local_obj_function[i] = pickle.load(input)
    with open('agent_{}_coup_function.pkl'.format(i), 'rb') as input:
        local_coup_function[i] = pickle.load(input)

# plot dual solutions
plt.figure()
plt.title("Dual solutions")
colors = {}
for i in range(N):
    colors[i] = np.random.rand(3, 1).flatten()
    dims = lambda_sequence[i].shape
    iterations = dims[0]
    for j in range(dims[1]):
        plt.plot(np.arange(iterations), lambda_sequence[i][:, j, 0], color=colors[i])

# plot cost of running average
plt.figure()
plt.title("Primal cost (running average)")

obj_function = np.zeros([iterations, 1])
for k in range(iterations):
    for i in range(N):
        obj_function[k] += local_obj_function[i].eval(xhat_sequence[i][k, :, 0].reshape(-1,1)).flatten()

plt.plot(obj_function)

# plot coupling constraint utilization
plt.figure()
plt.title("Coupling constraint utilization (running average)")

coup_function = np.zeros([iterations, S])
for k in range(iterations):
    for i in range(N):
        coup_function[k] += local_coup_function[i].eval(xhat_sequence[i][k, :, 0].reshape(-1,1)).flatten()

plt.plot(coup_function)

plt.show()
