import numpy as np
import matplotlib.pyplot as plt
import dill as pickle
from disropt.functions import Variable
from disropt.problems import LinearProblem

# initialize
with open('info.pkl', 'rb') as inp:
    info = pickle.load(inp)
NN        = info['N']
c         = info['c']
opt_sol   = info['opt_sol']

# load agent data
sequence_J = {}
for i in range(NN):
    sequence_J[i] = np.load("agent_{}_J_seq.npy".format(i))

# compute optimal cost
opt_cost = (c.transpose() @ opt_sol).flatten()

# plot cost evolution
plt.figure()
plt.title("Cost evolution")
colors = np.random.rand(NN+1, 3)
max_iters = 0

for i in range(NN):
    seq_J_i = sequence_J[i]
    n_iters_i = len(seq_J_i)
    max_iters = max(max_iters, n_iters_i)
    plt.plot(np.arange(n_iters_i), seq_J_i, color=colors[i])

# plot optimal cost
plt.plot(np.arange(max_iters), np.ones(max_iters)*opt_cost, '--', color=colors[NN])
plt.show()
