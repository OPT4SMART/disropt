import numpy as np
from ....agents import Agent
from .gradient_tracking import GradientTrackingTF


class GTAdamTF(GradientTrackingTF):
    def __init__(self, agent: Agent, beta1=0.9, beta2=0.999, epsilon=1e-8, enable_log: bool = False):
        super(GTAdamTF, self).__init__(agent, enable_log)

        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.G = 1e6
        self.m = None
        self.v = None

    def _update_local_solution(self, stepsize: float = 0.001):
        for i in range(len(self.x)):
            # update momenta
            self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * self.d[i]
            self.v[i] = np.minimum(self.beta2 * self.v[i] + (1 - self.beta2) * self.d[i] ** 2, self.G)
        
        for i in range(len(self.x)):
            # perform descent step
            direction = np.divide(self.m[i], np.sqrt(self.v[i]) + self.epsilon)
            self.x[i].assign_sub(stepsize * direction)

    def init_state(self):
        super().init_state()

        # initialize momenta
        self.m = [np.zeros_like(e) for e in self.x]
        self.v = [np.zeros_like(e) for e in self.x]
