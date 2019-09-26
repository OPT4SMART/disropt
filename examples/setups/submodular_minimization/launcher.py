from distoptpy.utils.submodular_func import stCutFn
from mpi4py import MPI
from distoptpy.agents import AwareAgent
from distoptpy.algorithms.gradient import SubgradientMethod
from distoptpy.utils.projection_sets import Box
from graph_constructor import binomial_random_graph, weights_dispenser
import imageio
import numpy as np
import matplotlib.pyplot as plt

comm = MPI.COMM_WORLD
nproc = comm.Get_size()
local_rank = comm.Get_rank()

# Graph generation
N = nproc
Adj = binomial_random_graph(N)
W = weights_dispenser(Adj)

# instantiate agent
agent = AwareAgent(in_neighbors=np.nonzero(Adj[local_rank, :])[0].tolist(), out_neighbors=np.nonzero(Adj[:, local_rank])[0].tolist(), in_weights=W[local_rank, :].tolist())

# load image and convert to [0,1] array
img = imageio.imread("./mickey32.jpg", as_gray=True)
img = img/255  # convert to 0 - 1

D = img.shape[0]
for ii in range(D):
    for jj in range(D):
        if img[ii, jj] > 0.9:
            img[ii, jj] = 1
        else:
            img[ii, jj] = 0


inf_var = 9999
DD = D*D
dist_plot = {}

# add noise to image
# TODO: corrupt only local portion of the image
prob_img = np.zeros([D, D])
prob_noise = 0.05
for ll in range(D):
    for kk in range(D):
        rand_num = np.random.rand()
        if rand_num > 1-prob_noise:
            prob_img[ll, kk] = 1-img[ll, kk]
        else:
            prob_img[ll, kk] = img[ll, kk]

i = agent.id
prob_img_node = 0.5*np.ones([D, D])
x_start = 0
x_end = 0
y_start = 0
y_end = 0
if i == 0:
    x_start += 0
    x_end += 16
    y_start += 0
    y_end += 32
elif i == 1:
    x_start += 12
    x_end += 32
    y_start += 0
    y_end += 32

prob_img_node[x_start:x_end,
              y_start:y_end] = prob_img[x_start:x_end, y_start:y_end]
prob_img_node += 0.00001*np.ones(img.shape)
pixel_prob_ = np.reshape(prob_img_node, (DD, 1))
pixel_prob = pixel_prob_ / (np.max(pixel_prob_)+0.0001)
prob_img_node = np.reshape(pixel_prob, (D, D))
pixel_res = -1*pixel_prob + 1
mu_r = np.zeros(pixel_prob.shape)
mu_back = np.zeros(pixel_res.shape)
for l in range(DD):
    if abs(pixel_prob[l]-0.5) > 0.1:
        mu_r[l] = -np.log(pixel_prob[l])
        mu_back[l] = -np.log(pixel_res[l])

# create s-t min-cut matrix
A = np.zeros([DD, DD])
lam = 0.5
beta = 5.0
for ii in range(x_start, x_end):
    for jj in range(y_start, y_end):
        # evaluate neighs [hardcoded]!
        neighs = np.array([[ii-1, jj-1],
                           [ii-1, jj],
                           [ii-1, jj+1],
                           [ii, jj-1],
                           [ii, jj+1],  # above-below
                           [ii+1, jj-1],
                           [ii+1, jj],
                           [ii+1, jj+1]])  # right
        # remove neighs out of img
        to_remove = []
        for kk in range(len(neighs)):
            if neighs[kk, 0] < 0:
                to_remove.append(kk)
            if neighs[kk, 1] < 0:
                to_remove.append(kk)
            if neighs[kk, 0] > D-1:
                to_remove.append(kk)
            if neighs[kk, 1] > D-1:
                to_remove.append(kk)

        to_remove = np.unique(to_remove)
        neighs = neighs[np.setdiff1d(range(8), to_remove), :]
        adj_i = ii+(jj-1)*D
        for ll in range(len(neighs)):
            adj_j = neighs[ll, 0]+(neighs[ll, 1]-1)*D
            adj_val = np.exp(-beta*((prob_img[ii, jj] - prob_img[neighs[ll, 0], neighs[ll, 1]])**2))
            A[adj_i, adj_j] = adj_val


Adj_stcut = np.zeros([DD+2, DD+2])
Adj_stcut[0:DD, 0:DD] = A
Adj_stcut[0:DD, -1] = lam*mu_r.flatten()
Adj_stcut[-2, 0:DD] = lam*mu_back.flatten()

fn = stCutFn(DD,Adj_stcut)
agent.set_objective_function(fn)
lb = np.zeros((DD, 1))
ub = np.ones((DD, 1))
box_constr = Box(lb,ub)
agent.add_constraint(box_constr) 

initial_condition = np.random.rand(DD,1)
algorithm = SubgradientMethod(agent, initial_condition, enable_log=True)

sequence = algorithm.run(iterations=10, stepsize=0.01)
print("Agent {}: {}".format(agent.id, algorithm.get_result().flatten()))

np.save("agents.npy", N)
np.save("agent_{}_sequence.npy".format(agent.id), sequence)

fig, axs = plt.subplots(nrows=1, ncols=1)
aux = algorithm.x
x_opt = np.reshape(aux, (D, D))
axs.imshow(x_opt,  cmap=plt.get_cmap('gray'))
plt.show()