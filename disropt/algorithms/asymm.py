import numpy as np
from copy import deepcopy
from scipy import stats
import time
import warnings
import random
from typing import Optional, List, Tuple, Any
from ..agents import Agent
from .algorithm import Algorithm
from .misc import AsynchronousLogicAnd, LogicAnd
from ..utils.utilities import is_pos_def
from ..functions import Variable, SquaredNorm, Square, Max, Norm, Square


class ASYMM(AsynchronousLogicAnd):
    """Asyncrhonous Distributed Method of Multipliers [FaGa19b]_

    See [FaGa19b]_ for the details.

    .. warning::
        This algorithm is currently under development

    """

    def __init__(self,
                 agent: Agent,
                 graph_diameter: int,
                 initial_condition: np.ndarray,
                 enable_log: bool = False,
                 **kwargs):
        super(ASYMM, self).__init__(agent=agent,
                                    graph_diameter=graph_diameter,
                                    enable_log=enable_log,
                                    **kwargs)

        self.x0 = initial_condition
        self.x_old = initial_condition
        self.x = initial_condition

        self.shape = self.x.shape

        self.x_neigh = {}

        # multipliers
        self.nu_from = {}
        self.nu_to = {}
        self.rho_from = {}
        self.rho_to = {}
        self.gr_cn = {}
        self.multiplier = {}
        self.penalty = {}
        self.constr_val = {}

        # Algorithm parameters
        # max penalty
        self.max_penalty = 1e5
        # constant stepsize
        self.alpha = 0.1
        # initial tolerance
        self.e = 5
        # penalty threshold
        self.gamma = 0.25
        # penalty growth parameter
        self.beta = 4

        # auxiliary variables
        self.M_done = False

        # local lagrangian
        self.local_lagrangian = None

    def __initialize_multipliers(self):
        for neigh in self.agent.in_neighbors:
            self.nu_from[neigh] = 0.01*np.random.rand(*self.shape)
            self.rho_from[neigh] = 1
            self.nu_to[neigh] = 0.01*np.random.rand(*self.shape)
            self.rho_to[neigh] = 1

            self.gr_cn[neigh] = np.linalg.norm(self.x - self.x_neigh[neigh], ord=2)
        for idx, constraint in enumerate(self.agent.problem.constraints):
            self.multiplier[idx] = 0.01*np.random.rand(*constraint.output_shape)
            self.penalty[idx] = 1

            if constraint.is_equality:
                self.constr_val[idx] = np.linalg.norm(constraint.function.eval(self.x), ord=2)
            elif constraint.is_inequality:
                self.constr_val[idx] = np.linalg.norm(
                    np.max([-self.multiplier[idx]/self.penalty[idx], constraint.function.eval(self.x)], axis=0))

    def __update_local_lagrangian(self):
        x = Variable(self.shape[0])

        # objective function
        fn = self.agent.problem.objective_function

        # neighboring constraints
        for j in self.agent.in_neighbors:
            fn += (self.nu_to[j] - self.nu_from[j]) @ x +\
                (self.rho_to[j]+self.rho_from[j]) / 2 * (x - self.x_neigh[j])@(x - self.x_neigh[j])

        # equality constraints
        for idx, constraint in enumerate(self.agent.problem.constraints):
            if constraint.is_equality:
                fn += self.multiplier[idx] @ constraint.function + \
                    0.5 * self.penalty[idx] * SquaredNorm(constraint.function)
            elif constraint.is_inequality:
                shape = constraint.output_shape
                fn += 1/(2*self.penalty[idx]) * np.ones(shape) @ (
                    Square(Max(0, self.multiplier[idx] + self.penalty[idx]*constraint.function)) - np.power(self.multiplier[idx], 2))

        self.local_lagrangian = fn

    def __update_stepsize(self):
        # m = 25
        # n = 10
        # delta = 1
        # space_diam = 5
        # l = np.zeros([m, 1])
        # for itr in range(m):
        #     for _ in range(n):
        #         p1 = self.x + space_diam * (np.random.rand(self.shape[0], self.shape[1])-0.5)
        #         p2 = p1 + delta * (np.random.rand(self.shape[0], self.shape[1])-0.5)
        #         dp = np.linalg.norm(p1-p2)
        #         while dp > delta:
        #             p2 = p1 + delta * (np.random.rand(self.shape[0], self.shape[1])-0.5)
        #             dp = np.linalg.norm(p1-p2)

        #         g1 = self.local_lagrangian.subgradient(p1)
        #         g2 = self.local_lagrangian.subgradient(p2)
        #         dg = np.linalg.norm(g1-g2)
        #         l[itr] = max(l[itr], dg/dp)

        # par = stats.weibull_min.fit(l)
        # self.alpha = 1/abs(par[-2])
        # TODO stepsize estimate
        self.alpha /= 1.2

    def __perform_descent(self, stepsize_type=None):
        if stepsize_type == 'newton':
            gradient = self.local_lagrangian.subgradient(self.x)
            hessian = self.local_lagrangian.hessian(self.x)
            if is_pos_def(hessian):
                #self.x -= 0.5 * np.linalg.inv(hessian) @ gradient
                self.x -= 0.5 * 1/np.linalg.norm(hessian) * gradient
            else:
                self.x -= min(self.alpha, self.e/10) * gradient
        else:
            self.x -= self.alpha * self.local_lagrangian.subgradient(self.x)

    def primal_update_step(self):
        self.__perform_descent()
        gradient = self.local_lagrangian.subgradient(self.x)
        if np.linalg.norm(gradient, ord=2) <= self.e:
            self.change_flag(new_flag=True)
        self.matrix_update()

    def dual_update_step(self):
        for neigh in self.agent.out_neighbors:
            self.nu_to[neigh] += self.rho_to[neigh]*(self.x - self.x_neigh[neigh])
            if np.linalg.norm(
                    self.x - self.x_neigh[neigh]) >= 0.01 and np.linalg.norm(
                    self.x - self.x_neigh[neigh]) >= self.gamma * self.gr_cn[neigh]:
                self.rho_to[neigh] = min(self.beta * self.rho_to[neigh], self.max_penalty)
            # norm of the constraint value
            self.gr_cn[neigh] = np.linalg.norm(self.x-self.x_neigh[neigh])

        for idx, constr in enumerate(self.agent.problem.constraints):
            if constr.is_equality:
                self.multiplier[idx] += self.penalty[idx] * constr.function.eval(self.x)
                if np.linalg.norm(constr.function.eval(self.x), ord=2) > self.gamma * self.constr_val[idx]:
                    self.penalty[idx] = min(self.beta * self.penalty[idx], self.max_penalty)
                self.constr_val[idx] = np.linalg.norm(constr.function.eval(self.x), ord=2)
                print(np.linalg.norm(constr.function.eval(self.x), ord=2))
            elif constr.is_inequality:
                self.multiplier[idx] = np.max(
                    [np.zeros(constr.output_shape),
                     self.multiplier[idx] + self.penalty[idx] * constr.function.eval(self.x)],
                    axis=0)
                if np.linalg.norm(
                    np.max([-self.multiplier[idx] / self.penalty[idx],
                            constr.function.eval(self.x)],
                           axis=0)) > self.gamma * self.constr_val[idx]:
                    self.penalty[idx] = min(self.beta * self.penalty[idx], self.max_penalty)
                self.constr_val[idx] = np.linalg.norm(
                    np.max([-self.multiplier[idx]/self.penalty[idx], constr.function.eval(self.x)], axis=0))

        self.M_done = True

    def reset_step(self):
        """
        Reset the matrix S and update e
        """
        self.M_done = False
        self.matrix_reset()
        self.e = self.e/1.5
        self.__update_stepsize()

    def run(self, running_time: float = 10):
        if not isinstance(running_time, float):
            raise TypeError("Running time must be a float")
        if self.enable_log:
            dims = [1]
            for dim in self.x.shape:
                dims.append(dim)
            self.dims = dims
            self.sequence = np.zeros(dims)
            self.sequence[0] = self.x
            self.timestamp_sequence_awake = [time.time()]
            self.timestamp_sequence_sleep = [time.time()]

        # Exchange all data at the beginning
        data = self.agent.neighbors_exchange(self.x)
        for neigh in data:
            self.x_neigh[neigh] = data[neigh]
        self.__initialize_multipliers()
        self.__update_local_lagrangian()
        self.__update_stepsize()

        received_nu = 0
        end_time = time.time() + running_time
        while time.time() < end_time:
            if self.enable_log:
                self.timestamp_sequence_awake.append(time.time())

            data = self.agent.neighbors_receive_asynchronous()
            for neigh in data:
                msg = data[neigh]
                if msg["type"] == "primal":
                    self.x_neigh[neigh] = msg['x']
                    self.update_column(neigh, msg['S'])
                elif msg["type"] == "dual":
                    self.force_matrix_update()
                    self.nu_from[neigh] = msg['nu']
                    self.rho_from[neigh] = msg['rho']
                    received_nu += 1

            if self.M_done and received_nu == len(self.agent.in_neighbors):
                received_nu = 0
                self.reset_step()
                self.__update_local_lagrangian()

            if not self.check_stop() and not self.M_done:
                self.primal_update_step()
                msg = {'type': "primal",
                       'x': self.x,
                       'S': self.S[:, -1]}
                self.agent.neighbors_send(msg)

            if self.check_stop() and not self.M_done:
                self.dual_update_step()
                for neigh in self.agent.out_neighbors:
                    msg = {'type': "dual",
                           'nu': self.nu_to[neigh],
                           'rho': self.rho_to[neigh]}
                    self.agent.communicator.neighbors_send(msg, [neigh])

            # save sequence
            if self.enable_log:
                self.sequence = np.vstack([self.sequence, self.x.reshape(self.dims)])

            if self.enable_log:
                self.timestamp_sequence_sleep.append(time.time())

        if self.enable_log:
            return self.timestamp_sequence_awake, \
                self.timestamp_sequence_sleep, \
                self.sequence

    def get_result(self):
        """Return the value of the solution
        """
        return self.x
