import dill as pickle
import numpy as np
import matplotlib.pyplot as plt

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

# plot sequence of x
plt.figure()
plt.title("Primal sequences")

colors = {}
for i in range(N):
    colors[i] = np.random.rand(3, 1).flatten()
    dims = x_sequence[i].shape
    print(dims)
    iterations = dims[0]
    for j in range(dims[1]):
        plt.plot(np.arange(iterations), x_sequence[i][:, j, 0], color=colors[i])

# plot primal cost
plt.figure()
plt.title("Primal cost")

function = np.zeros([iterations, 1])
for k in range(iterations):
    avg = np.zeros([n, 1])
    for i in range(N):
        avg += x_sequence[i][k, :, 0].reshape(n, 1)
    avg = avg/N
    for i in range(N):
        function[k] += local_obj_function[i].eval(avg).flatten()

plt.plot(np.arange(iterations), function)

plt.show()
