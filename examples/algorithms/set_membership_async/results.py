import numpy as np
import matplotlib.pyplot as plt
from matplotlib2tikz import save as tikzsave


N = np.load("agents.npy")

# retrieve local sequences
sequence = {}
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

for i in range(N):
    timestamp_sequence_awake[i] = timestamp_sequence_awake[i] - t_init
    timestamp_sequence_sleep[i] = timestamp_sequence_sleep[i] - t_init

plt.figure()
colors = {}
for i in range(N):
    colors[i] = np.random.rand(3, 1).flatten()

    dims = sequence[i].shape
    for j in range(dims[1]):
        if j == 0:
            plt.plot(timestamp_sequence_sleep[i], sequence[i][:, j, 0],
                     color=colors[i],
                     label="Agent {}: awakenings={}".format(i+1, dims[0]))
        else:
            plt.plot(timestamp_sequence_sleep[i], sequence[i][:, j, 0],
                     color=colors[i])
plt.xlabel("time (s)")
plt.ylabel("x_i")
plt.legend()
tikzsave("estimates.tex")

S = {}
for i in range(N):
    aw = np.array(timestamp_sequence_awake[i])
    aw = np.vstack([aw, np.zeros(aw.shape)])
    sl = np.array(timestamp_sequence_sleep[i])
    sl = np.vstack([sl, np.ones(sl.shape)])
    aux = np.hstack([aw, sl]).transpose()
    signal = aux[aux[:, 0].argsort()]
    inverse_signal = np.zeros(signal.shape)
    inverse_signal += signal
    inverse_signal[:, 1] = abs(inverse_signal[:, 1] - 1)
    ww = np.empty([signal.shape[0]*2, 2])
    ww[::2] = signal
    ww[1::2] = inverse_signal
    S[i] = ww

fig, axes = plt.subplots(int(N), 1, sharex=True)
for i in range(N):
    axes[i].plot(S[i][:, 0], S[i][:, 1], color=colors[i])
    axes[i].set_xlim([0,3])
    axes[i].set_ylabel("Agent {}".format(i))
plt.xlabel("time (s)")
tikzsave("awakenings.tex")
plt.show()
