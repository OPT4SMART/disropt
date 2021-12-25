from copy import deepcopy
from ....agents import Agent
from .subgradient import SubgradientMethodTF


class GradientTrackingTF(SubgradientMethodTF): # structure of this class is similar to SubgradientMethodTF
    def __init__(self, agent: Agent, enable_log: bool = False):
        super(GradientTrackingTF, self).__init__(agent, enable_log)

        self.d = None
        self.grad_old = None

    def _update_local_solution(self, stepsize: float = 0.001):
        # perform gradient step with tracker
        for i in range(len(self.x)):
            self.x[i].assign_sub(stepsize * self.d[i])
    
    def _update_local_tracker(self):
        # compute new gradient
        grad_new = self.agent.problem.objective_function.subgradient()

        # update tracker
        for i in range(len(self.d)):
            self.d[i] += grad_new[i] - self.grad_old[i]
        
        # store new gradient
        self.grad_old = grad_new

    def iterate_run(self, **kwargs):
        """Run a single iterate of the algorithm
        """
        # exchange solution with neighbors
        neigh_x = self.agent.neighbors_exchange([t.numpy() for t in self.x])

        # perform consensus step on solution
        for i in range(len(self.x)):
            self.x[i].assign(self.agent.in_weights[self.agent.id] * self.x[i])

            for j, x_j in neigh_x.items():
                self.x[i].assign_add(self.agent.in_weights[j] * x_j[i])
        
        # update solution
        self._update_local_solution(**kwargs)

        # exchange tracker with neighbors
        neigh_d = self.agent.neighbors_exchange(self.d)

        # perform consensus step on tracker
        for i in range(len(self.d)):
            self.d[i] *= self.agent.in_weights[self.agent.id]
            
            for j, d_j in neigh_d.items():
                self.d[i] += self.agent.in_weights[j] * d_j[i]
        
        # update tracker
        self._update_local_tracker()

    def init_state(self):
        # initialize tracker and current gradient at x0
        self.loss.init_epoch()
        self.loss.load_batch()
        self.grad_old = self.loss.subgradient()
        self.d = deepcopy(self.grad_old)
