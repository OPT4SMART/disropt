import numpy as np
from disropt.agents import Agent
from disropt.algorithms.gradient_tracking import GradientTracking
from disropt.functions import QuadraticForm, Variable
from disropt.utils.graph_constructor import MPIgraph
from disropt.problems import Problem

# generate communication graph (everyone uses the same seed)
comm_graph = MPIgraph('random_binomial', 'metropolis')
agent_id, in_nbrs, out_nbrs, in_weights, _ = comm_graph.get_local_info()

# size of optimization variable
n = 5

# generate quadratic cost function
np.random.seed()
Q = np.random.randn(n, n)
Q = Q.transpose() @ Q
x = Variable(n)
func = QuadraticForm(x - np.random.randn(n, 1), Q)

# create Problem and Agent
agent = Agent(in_nbrs, out_nbrs, in_weights=in_weights)
agent.set_problem(Problem(func))

# run the algorithm
x0 = np.random.rand(n, 1)
algorithm = GradientTracking(agent, x0)
algorithm.run(iterations=1000, stepsize=0.01)

print("Agent {} - solution estimate: {}".format(agent_id, algorithm.get_result().flatten()))
