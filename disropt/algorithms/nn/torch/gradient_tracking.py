import torch
from copy import deepcopy
from ....agents import Agent
from .subgradient import SubgradientMethodTorch


class GradientTrackingTorch(SubgradientMethodTorch): # structure of this class is similar to SubgradientMethodTorch
    def __init__(self, agent: Agent, enable_log: bool = False):
        super(GradientTrackingTorch, self).__init__(agent, enable_log)

        self.d = None
        self.grad_old = None

    def _update_local_solution(self, stepsize: float = 0.001):
        # perform gradient step with tracker
        with torch.no_grad():
            k = 0
            
            for p_k in iter(self.x):
                if p_k.requires_grad:
                    p_k.add_(torch.tensor(self.d[k]), alpha=-stepsize)
                    k += 1
    
    def _update_local_tracker(self):
        # compute new gradient
        grad_new = [e.detach().numpy() for e in self.loss.subgradient(return_gradients=True)]

        # update tracker
        for i in range(len(self.d)):
            self.d[i] += grad_new[i] - self.grad_old[i]
        
        # store new gradient
        self.grad_old = grad_new

    def iterate_run(self, **kwargs):
        """Run a single iterate of the algorithm
        """
        with torch.no_grad():
            # exchange solution with neighbors
            my_x = [p_k for p_k in iter(self.x) if p_k.requires_grad] # exchange only trainable parameters
            neigh_x = self.agent.neighbors_exchange(my_x)

            # perform consensus step on solution
            k = 0
            for p_k in iter(self.x):
                if p_k.requires_grad: # perform only on only trainable parameters
                    p_k.mul_(self.agent.in_weights[self.agent.id])

                    for j, x_j in neigh_x.items():
                        p_k.add_(x_j[k], alpha=self.agent.in_weights[j])
                    
                    k += 1
        
        # update solution
        self._update_local_solution(**kwargs)

        with torch.no_grad():
            # exchange tracker with neighbors
            neigh_d = self.agent.neighbors_exchange(self.d)

            # perform consensus step on tracker
            for k, d_k in enumerate(iter(self.d)):
                d_k *= self.agent.in_weights[self.agent.id]

                for j, y_j in neigh_d.items():
                    d_k += self.agent.in_weights[j] * y_j[k]
        
        # update tracker
        self._update_local_tracker()

    def init_state(self):
        # initialize tracker and current gradient at x0
        self.loss.init_epoch()
        self.loss.load_batch()
        self.grad_old = [e.detach().numpy() for e in self.loss.subgradient(return_gradients=True)]
        self.d = deepcopy(self.grad_old)
