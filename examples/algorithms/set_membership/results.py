import numpy as np
import matplotlib.pyplot as plt

N = np.load("agents.npy")

sequence = {}
for i in range(N):
    filename = "agent_{}_sequence.npy".format(i)
    sequence[i] = np.load(filename)

plt.figure()
for i in range(N):
    dims = sequence[i].shape
    iterations = dims[0]
    for j in range(dims[1]):
        plt.plot(np.arange(iterations), sequence[i][:, j, 0])
plt.show()
