###############################################################
# COST-COUPLED Example
# Training of Neural Network for classification
#
# Each agent has a subset of training and test dataset (MNIST).
# Agents aim to train a (common) Neural Network by exploiting
# the whole distributed dataset.
###############################################################
# Used algorithm: Gradient Tracking
# Deep learning framework: PyTorch
###############################################################

##### PYTORCH ######
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor, Lambda

import dill as pickle
import numpy as np
import time
from mpi4py import MPI
from disropt.agents import Agent
from disropt.algorithms.nn.torch import GradientTrackingTorch
from disropt.functions.nn import TorchLoss
from disropt.utils.graph_constructor import binomial_random_graph, metropolis_hastings
from disropt.utils.torchnn import MyDistributedSampler, CategoricalAccuracyMetric, MeanMetric
from disropt.problems import Problem


# script configuration
epochs = 100

N_TRAINING_SAMP = 1500
N_TEST_SAMP = 1000
BATCH_SIZE = 10 # 5 agents -> 300 samples per agent -> 30 batches per agent
BATCH_SIZE_TEST = 10 # 5 agents -> 200 samples per agent -> 20 batches per agent

stepsize = 1e-3

#####################

# get MPI info
NN = MPI.COMM_WORLD.Get_size()
agent_id = MPI.COMM_WORLD.Get_rank()

# Generate a common graph (everyone uses the same seed)
Adj = binomial_random_graph(NN, p=0.2, seed=1)
W = metropolis_hastings(Adj)

np.random.seed(10*agent_id)
torch.manual_seed(10*agent_id)

device = "cuda" if torch.cuda.is_available() else "cpu"
if agent_id == 0:
    print(f"Using device {device}")


#####################
# Load MNIST dataset
#####################

# load training and test datasets
# labels are converted to one-hot
training_data = datasets.MNIST(
    root="data",
    train=True,
    download=True,
    transform=ToTensor(),
    target_transform=Lambda(lambda y: torch.zeros(10, dtype=torch.float).scatter_(0, torch.tensor(y), value=1))
)

test_data = datasets.MNIST(
    root="data",
    train=False,
    download=True,
    transform=ToTensor(),
    target_transform=Lambda(lambda y: torch.zeros(10, dtype=torch.float).scatter_(0, torch.tensor(y), value=1))
)

# split dataset among agents and initialize batched dataloader
train_sampler = MyDistributedSampler(training_data, NN, agent_id, n_samples=N_TRAINING_SAMP, shuffle=True, seed=10*agent_id)
test_sampler = MyDistributedSampler(test_data, NN, agent_id, n_samples=N_TEST_SAMP)
train_ds = DataLoader(training_data, batch_size=BATCH_SIZE, sampler=train_sampler)
test_ds = DataLoader(test_data, batch_size=BATCH_SIZE_TEST, sampler=test_sampler)

#####################
# Create Neural Network model
#####################

# create sequential model
model = nn.Sequential(
    nn.Conv2d(1, 32, kernel_size=5,padding='same'),
    nn.ReLU(),
    nn.MaxPool2d(2),
    # nn.Dropout(0.4),
    nn.Conv2d(32, 64, kernel_size=5,padding='same'),
    nn.ReLU(),
    nn.MaxPool2d(2),
    # nn.Dropout(0.4),
    nn.Flatten(),
    nn.Linear(3136,128),
    nn.ReLU(),
    # nn.Dropout(0.4),
    nn.Linear(128,10),
    nn.Softmax(dim=1)
).to(device)

# define model metrics
train_loss = MeanMetric()
train_accuracy = CategoricalAccuracyMetric()
test_accuracy = CategoricalAccuracyMetric()


#####################
# Epoch callback
#####################

# initialize test accuracy and consensus error sequences
test_accuracy_seq = np.zeros(epochs)
consensus_err_seq = np.zeros(epochs)

def callback_func(epoch):
    ### COMPUTE ACCURACY ###
    test_accuracy.reset_states()
    model.eval() # set model in evaluation mode
    for X, Y in test_ds:
        predictions = model(X)
        test_accuracy.update_state(Y, predictions)
    test_accuracy_seq[epoch] = test_accuracy.result()

    ### COMPUTE CONSENSUS ERROR ###
    result = 0
    weights_loc = model.parameters()

    # cycle over variables
    for p_k in weights_loc:
        w_k_loc = p_k.detach().numpy().astype('float32')
        w_k_mean = np.zeros_like(w_k_loc, dtype='float32')

        # perform MPI Allreduce to compute average of k-th variable
        MPI.COMM_WORLD.Allreduce([w_k_loc, MPI.FLOAT], [w_k_mean, MPI.FLOAT], op=MPI.SUM)
        w_k_mean /= NN
        
        # accumulate consensus error of k-th variable
        result += np.linalg.norm((w_k_loc - w_k_mean).flatten())
    
    # save
    consensus_err_seq[epoch] = result


#####################
# Train model
#####################

tstart = time.time()

# create local agent
agent = Agent(in_neighbors=np.nonzero(Adj[agent_id, :])[0].tolist(),
              out_neighbors=np.nonzero(Adj[:, agent_id])[0].tolist(),
              in_weights=W[agent_id, :].tolist())

# create local problem
loss_fn = nn.MSELoss()
obj_func = TorchLoss(model, train_ds, loss_fn, device)
obj_func.set_metrics(train_loss, train_accuracy)
prob = Problem(objective_function=obj_func)
agent.set_problem(prob)

# initialize algorithm
algorithm = GradientTrackingTorch(agent, enable_log=True)

train_loss_seq, train_acc_seq, time_seq = algorithm.run(
        epochs=epochs,
        stepsize=stepsize,
        callback_func=callback_func
    )


#####################
# Save results
#####################

if agent_id == 0:
    print('Saving result')

# save algorithm results
result_alg = {
        'train_loss': train_loss_seq,
        'train_accuracy': train_acc_seq,
        'test_accuracy': test_accuracy_seq,
        'consensus_err': consensus_err_seq,
        'time': time_seq
    }

with open('result_{}.pkl'.format(agent_id), 'wb') as output:
    pickle.dump(result_alg, output, pickle.HIGHEST_PROTOCOL)

# save number of agents and epochs
if agent_id == 0:
    with open('info.pkl', 'wb') as output:
        pickle.dump({'N': NN, 'epochs': epochs}, output, pickle.HIGHEST_PROTOCOL)
    
    tend = time.time()
    print('Total elapsed time: {} seconds'.format(tend - tstart))
