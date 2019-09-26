import numpy as np

def binomial_random_graph(N):
    np.random.seed(1)
    while True:
        p = 0.3
        Adj = np.random.binomial(1, p, (N, N))  # each entry is 1 with prob "p"

        Adj = np.logical_or(Adj, Adj.transpose()) # Undirected
        I_NN = np.eye(N)

        # Adj = np.logical_or(Adj, I_NN).astype(int)  # add self - loops
        Adj = np.logical_and(Adj, np.logical_not(I_NN)).astype(int)  # remove self - loops
        testAdj = np.linalg.matrix_power((I_NN + Adj), N)  # check if G is connected

        check = 0
        for i in range(N):
            check += int(bool(len(np.nonzero(Adj[:, i])[0]))) + int(bool(len(np.nonzero(Adj[i])[0])))

        if not np.any(np.any(np.logical_not(testAdj))) and check == 2*N:
            break
    return Adj

def weights_dispenser(Adj):
    # eighted adjacency matrix (Metropolis-Hastings)
    N = np.shape(Adj)[0]
    degree = np.sum(Adj, axis=0)
    W = np.zeros([N, N]) 
    for i in range(N):
        N_i = np.nonzero(Adj[i, :])[0] # Fixed Neighbors
        for j in N_i:
            W[i,j] = 1/(1+np.max([degree[i], degree[j]]))
        W[i,i] = 1 - np.sum(W[i, :])
    return W
