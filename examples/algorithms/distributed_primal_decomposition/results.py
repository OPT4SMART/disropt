import numpy as np
import matplotlib.pyplot as plt
import pickle

N = np.load("agents.npy")
S = 3

y_sequence = {}
x_sequence = {}
local_obj_function = {}
local_coup_function = {}
for i in range(N):
    y_sequence[i] = np.load("agent_{}_allocation_sequence.npy".format(i))
    x_sequence[i] = np.load("agent_{}_primal_sequence.npy".format(i))
    with open('agent_{}_obj_function.pkl'.format(i), 'rb') as input:
        local_obj_function[i] = pickle.load(input)
    with open('agent_{}_coup_function.pkl'.format(i), 'rb') as input:
        local_coup_function[i] = pickle.load(input)

# plot cost of running average
plt.figure()
plt.title("Primal cost")

iterations = x_sequence[i].shape[0]
obj_function = np.zeros([iterations, 1])
for k in range(iterations):
    for i in range(N):
        obj_function[k] += local_obj_function[i].eval(x_sequence[i][k, :, 0].reshape(-1,1)).flatten()

plt.semilogy(obj_function)

# plot coupling constraint utilization
plt.figure()
plt.title("Coupling constraint utilization")

coup_function = np.zeros([iterations, S])
for k in range(iterations):
    for i in range(N):
        coup_function[k] += local_coup_function[i].eval(x_sequence[i][k, :, 0].reshape(-1,1)).flatten()

plt.plot(coup_function)

plt.show()
