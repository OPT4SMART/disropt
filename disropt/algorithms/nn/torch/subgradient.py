import numpy as np
import time
import torch
from typing import Callable
from ....agents import Agent
from ....functions.nn import TorchLoss
from ...algorithm import Algorithm


class SubgradientMethodTorch(Algorithm):
    def __init__(self, agent: Agent, enable_log: bool = False):
        if not isinstance(agent, Agent):
            raise TypeError("Agent must be an Agent")
        if not isinstance(agent.problem.objective_function, TorchLoss):
            raise TypeError("The agent must be equipped with a TorchLoss objective function")
        super(SubgradientMethodTorch, self).__init__(agent, enable_log)

        self.loss = agent.problem.objective_function
        self.x = list(self.loss.get_model_parameters())
        self.loss_sequence = None
        self.acc_sequence  = None
        self.time_sequence = None

    def _update_local_solution(self, stepsize: float = 0.001):
        # perform gradient step
        self.loss.subgradient()

        with torch.no_grad():
            for p_k in iter(self.x):
                if p_k.grad is not None:
                    p_k.add_(p_k.grad, alpha=-stepsize)

    def iterate_run(self, **kwargs):
        """Run a single iterate of the algorithm
        """
        with torch.no_grad():
            # exchange data with neighbors
            my_x = [p_k for p_k in iter(self.x) if p_k.requires_grad] # exchange only trainable parameters
            neigh_x = self.agent.neighbors_exchange(my_x)

            # perform consensus step
            k = 0
            for p_k in iter(self.x):
                if p_k.requires_grad: # perform only on only trainable parameters
                    p_k.mul_(self.agent.in_weights[self.agent.id])

                    for j, x_j in neigh_x.items():
                        p_k.add_(x_j[k], alpha=self.agent.in_weights[j])
                    
                    k += 1

        self._update_local_solution(**kwargs)

    def run(self, epochs: int = 1000, stepsize = 0.001, callback_func: Callable = None):
        if not isinstance(epochs, int):
            raise TypeError("The number of epochs must be an int")
        if not (isinstance(stepsize, float) or callable(stepsize)):
            raise TypeError("The stepsize must be a float or a function")
        
        self.loss_sequence = np.zeros(epochs)
        self.acc_sequence  = np.zeros(epochs)
        self.time_sequence = np.zeros(epochs)

        self.init_state()

        for epoch in range(epochs):

            if not isinstance(stepsize, float):
                step = stepsize(epoch)
            else:
                step = stepsize
            
            if self.enable_log:
                self.loss.reset_metrics()

            time_start = time.time()
            self.loss.init_epoch()

            while self.loss.load_batch():
                self.iterate_run(stepsize=step)
            
            time_end = time.time()

            if self.enable_log:
                loss, acc = self.loss.metrics_result()
                self.loss_sequence[epoch] = loss
                self.acc_sequence[epoch]  = acc
                self.time_sequence[epoch] = time_end - time_start
            
            if callback_func is not None:
                callback_func(epoch)
            
            print('Epoch {}'.format(epoch), end='\r')
        
        if self.enable_log:
            return self.loss_sequence, self.acc_sequence, self.time_sequence

    def init_state(self):
        pass # do nothing (method will be overridden by subclasses)
