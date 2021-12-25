###############################################################
# COST-COUPLED Example
# Training of Neural Network for classification
#
# Each agent has a subset of training and test dataset (MNIST).
# Agents aim to train a (common) Neural Network by exploiting
# the whole distributed dataset.
###############################################################
# Used algorithm: GTAdam
# Deep learning framework: Tensorflow
###############################################################

##### TENSORFLOW ######
import tensorflow as tf
from tensorflow import keras

import dill as pickle
import numpy as np
import time
from mpi4py import MPI
from disropt.agents import Agent
from disropt.algorithms.nn.tensorflow import GTAdamTF
from disropt.functions.nn import TensorflowLoss
from disropt.utils.graph_constructor import binomial_random_graph, metropolis_hastings
from disropt.problems import Problem


# script configuration
epochs = 50

N_TRAINING_SAMP = 1500
N_TEST_SAMP = 1000
BATCH_SIZE = 10 # 5 agents -> 300 samples per agent -> 30 batches per agent
BATCH_SIZE_TEST = 10 # 5 agents -> 200 samples per agent -> 20 batches per agent

stepsize = 1e-5

#####################

# get MPI info
NN = MPI.COMM_WORLD.Get_size()
agent_id = MPI.COMM_WORLD.Get_rank()

# Generate a common graph (everyone uses the same seed)
Adj = binomial_random_graph(NN, p=0.2, seed=1)
W = metropolis_hastings(Adj)

np.random.seed(10*agent_id)
tf.random.seed(10*agent_id)


#####################
# Load MNIST dataset
#####################

# load full dataset
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data() # in ~/.keras/datasets

# add a channels dimension
x_train = x_train.reshape((60000, 28, 28, 1)).astype('float32')
x_test = x_test.reshape((10000, 28, 28, 1)).astype('float32')

# convert labels to one-hot tensors
y_train = tf.one_hot(y_train, 10, dtype=tf.int32)
y_test = tf.one_hot(y_test, 10, dtype=tf.int32)

# form training and test dataset
train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train)).take(N_TRAINING_SAMP).shard(NN, agent_id)
test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test)).take(N_TEST_SAMP)

# normalize data
train_ds = train_ds.map(lambda x, y: (x / 255.0, y))
test_ds  = test_ds.map(lambda x, y: (x / 255.0, y))

# shuffle and create batches
train_ds = train_ds.shuffle(int(N_TRAINING_SAMP / NN)).batch(BATCH_SIZE)
test_ds = test_ds.batch(BATCH_SIZE_TEST)


#####################
# Create Neural Network model
#####################

# create model using Keras
model = keras.Sequential()
model.add(keras.layers.Conv2D(32,kernel_size=5,padding='same',activation='relu', input_shape=(28,28,1)))
model.add(keras.layers.MaxPool2D())
# model.add(keras.layers.Dropout(0.4))
model.add(keras.layers.Conv2D(64,kernel_size=5,padding='same',activation='relu'))
model.add(keras.layers.MaxPool2D())
# model.add(keras.layers.Dropout(0.4))
model.add(keras.layers.Flatten())
model.add(keras.layers.Dense(128, activation='relu'))
# model.add(keras.layers.Dropout(0.4))
model.add(keras.layers.Dense(10, activation='softmax'))

# define model metrics
train_loss = tf.keras.metrics.Mean(name='train_loss')
train_accuracy = tf.keras.metrics.CategoricalAccuracy(name='train_accuracy')
test_accuracy = tf.keras.metrics.CategoricalAccuracy(name='test_accuracy')


#####################
# Epoch callback
#####################

# initialize test accuracy and consensus error sequences
test_accuracy_seq = np.zeros(epochs)
consensus_err_seq = np.zeros(epochs)

# compiled helper function for computing test accuracy
@tf.function
def _compute_accuracy(model):
    for X, Y in test_ds:
        predictions = model(X, training=False)
        test_accuracy(Y, predictions)

def callback_func(epoch):
    ### COMPUTE ACCURACY ###
    test_accuracy.reset_states()
    _compute_accuracy(model)
    test_accuracy_seq[epoch] = test_accuracy.result()

    ### COMPUTE CONSENSUS ERROR ###
    result = 0
    weights_loc = model.trainable_variables

    # cycle over variables
    for k in range(len(weights_loc)):
        w_k_loc = tf.convert_to_tensor(weights_loc[k]).numpy().astype('float32')
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
obj_func = TensorflowLoss(model, train_ds, tf.keras.losses.MSE)
obj_func.set_metrics(train_loss, train_accuracy)
prob = Problem(objective_function=obj_func)
agent.set_problem(prob)

# initialize algorithm
algorithm = GTAdamTF(agent, enable_log=True)

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
