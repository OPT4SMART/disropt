import numpy as np
import dill as pickle
import matplotlib.pyplot as plt


# number of agents
N = np.load("agents.npy")

# retrieve local sequences
sequence = {}
constraints = {}
timestamp_sequence_awake = {}
timestamp_sequence_sleep = {}
colors = {}
t_init = None
for i in range(N):
    colors[i] = np.random.rand(3, 1).flatten()
    filename = "agent_{}_sequence.npy".format(i)
    sequence[i] = np.load(filename, allow_pickle=True)

    filename = "agent_{}_timestamp_sequence_awake.npy".format(i)
    timestamp_sequence_awake[i] = np.load(filename, allow_pickle=True)

    filename = "agent_{}_timestamp_sequence_sleep.npy".format(i)
    timestamp_sequence_sleep[i] = np.load(filename, allow_pickle=True)

    if t_init is not None:
        m = min(timestamp_sequence_awake[i])
        t_init = min(t_init, m)
    else:
        t_init = min(timestamp_sequence_awake[i])

    with open('agent_{}_constraints.pkl'.format(i), 'rb') as input:
        constraints[i] = pickle.load(input)

for i in range(N):
    timestamp_sequence_awake[i] = timestamp_sequence_awake[i] - t_init
    timestamp_sequence_sleep[i] = timestamp_sequence_sleep[i] - t_init

# plot
plt.figure()
for i in range(N):
    dims = sequence[i].shape
    for j in range(dims[1]):
        for m in range(dims[2]):
            plt.plot(timestamp_sequence_sleep[i], sequence[i][:, j, m])
plt.ylim([-5, 5])

# # plt
plt.figure()
for i in range(N):
    dims = sequence[i].shape
    iterations = dims[0]
    print("Agent {}, iterations {}".format(i, iterations))
    feasibility = np.zeros([iterations, 2])
    for k in range(iterations):
        for idx, constr in enumerate(constraints[i]):
            flag = constr.eval((sequence[i][k, :, :]).reshape(2,1))
            if flag == False:
                feasibility[k, idx] += abs(constr.function.eval((sequence[i][k, :, :]).reshape(2,1)).flatten())
    plt.semilogy(timestamp_sequence_sleep[i], feasibility)
# plt.lim([-1, 15])

plt.show()
