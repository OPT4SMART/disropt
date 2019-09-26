import numpy as np
import matplotlib.pyplot as plt


# number of agents
N = np.load("agents.npy")

# retrieve local sequences
sequence = {}
timestamp_sequence_awake = {}
timestamp_sequence_sleep = {}
colors = {}
for i in range(N):
    colors[i] = np.random.rand(3, 1).flatten()
    filename = "agent_{}_sequence.npy".format(i)
    sequence[i] = np.load(filename, allow_pickle=True)

    filename = "agent_{}_timestamp_sequence_awake.npy".format(i)
    timestamp_sequence_awake[i] = np.load(filename, allow_pickle=True)

    filename = "agent_{}_timestamp_sequence_sleep.npy".format(i)
    timestamp_sequence_sleep[i] = np.load(filename, allow_pickle=True)

# plot
plt.figure()
for i in range(N):
    dims = sequence[i].shape
    for j in range(dims[1]):
        for m in range(dims[2]):
            plt.plot(timestamp_sequence_sleep[i], sequence[i][:, j, m])

plt.show()
