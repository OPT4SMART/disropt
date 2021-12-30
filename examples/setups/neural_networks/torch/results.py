import numpy as np
import matplotlib.pyplot as plt
import dill as pickle


# initialize
with open('info.pkl', 'rb') as inp:
    info = pickle.load(inp)
NN     = info['N']
epochs = info['epochs']

# load agent data
agents = {}
train_accuracy_seq = np.zeros((epochs, NN))

for i in range(NN):
    agents[i] = {}
    with open('result_{}.pkl'.format(i), 'rb') as inp:
        agents[i] = pickle.load(inp)

    train_accuracy_seq[:, i] = agents[i]['train_accuracy']

time_seq = agents[0]['time']
test_accuracy_seq = np.mean([agents[i]['test_accuracy'] for i in range(NN)], axis=0)
train_loss_seq    = np.mean([agents[i]['train_loss'] for i in range(NN)], axis=0)
consensus_err_seq = np.sum([agents[i]['consensus_err'] for i in range(NN)], axis=0)

####################################

# plot computation time
plt.figure()
plt.title('Computation time per epoch')
plt.xlabel(r"epochs")
plt.ylabel(r"elapsed time")
plt.plot(np.arange(epochs), time_seq)

# plot test accuracy
plt.figure()
plt.title('Test accuracy over epochs')
plt.xlabel(r"epochs")
plt.ylabel(r"test accuracy")
plt.plot(np.arange(epochs), test_accuracy_seq*100)

# plot training accuracy
plt.figure()
plt.title('Training accuracy over epochs')
plt.xlabel(r"epochs")
plt.ylabel(r"train accuracy")
plt.plot(np.arange(epochs), train_accuracy_seq*100)
plt.legend()

# plot loss
plt.figure()
plt.title('Training loss over epochs')
plt.xlabel(r"epochs")
plt.ylabel(r"loss value")
plt.plot(np.arange(epochs), train_loss_seq)

# plot consensus error
plt.figure()
plt.title('Consensus error over epochs')
plt.xlabel(r"epochs")
plt.ylabel(r"consensus error")
plt.plot(np.arange(epochs), consensus_err_seq)

plt.show()