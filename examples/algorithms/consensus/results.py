import numpy as np
import matplotlib.pyplot as plt

# Load the number of agents
N = np.load("agents.npy")

# Load the locally generated sequences
sequence = {}
for i in range(N):
    filename = "agent_{}_sequence.npy".format(i)
    sequence[i] = np.load(filename)

# Plot the evolution of the local estimates
# generate N colors
colors = {}
for i in range(N):
    colors[i] = np.random.rand(3, 1).flatten()
# generate figure
plt.figure()
for i in range(N):
    dims = sequence[i].shape
    iterations = dims[0]
    for j in range(dims[1]):
        if j == 0: # to avoid generating multiple labels
            plt.plot(np.arange(iterations), sequence[i][:, j, 0], color=colors[i], label='agent {}'.format(i+1))
        else:
            plt.plot(np.arange(iterations), sequence[i][:, j, 0], color=colors[i])
plt.legend(loc='upper right')
plt.title("Evolution of the local estimates")
plt.xlabel(r"iterations ($k$)")
plt.ylabel(r"$x_i^k$")
plt.savefig('results_fig.png')
plt.show()
