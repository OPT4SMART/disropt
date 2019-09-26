import numpy as np
import time
from copy import deepcopy
from typing import Union, Callable
from ..agents import Agent
from ..constraints import AbstractSet, Constraint
from ..problems import ProjectionProblem
from .consensus import Consensus, AsynchronousConsensus


class SetMembership(Consensus):
    """Distributed Set Membership Algorithm [FaGa18]_

    From the perspective of agent :math:`i` the algorithm works as follows. For :math:`k=0,1,\\dots`

    .. math::

        X_i^{k+1} &= X_i^k \cap M_i^{k+1}

        z_i^{k} &= \sum_{j=1}^N w_{ij} x_j^k

        x_i^{k+1} &= \Pi_{X_i^{k+1}}[z_i^k]

    where :math:`x_i,z_i\\in\\mathbb{R}^n`, :math:`M_i^{k+1},X_i^{k+1}\\subseteq\mathbb{R}^n` are the current feasible (measurement) set and the feasible (parameter) set respectively, and :math:`\\Pi_X[]` denotes the projection operator over the set :math:`X`.


    Args:
        agent (Agent): agent to execute the algorithm (must be a Agent)
        initial_condition (numpy.ndarray): initial condition
        enable_log (bool): True for enabling log

    Attributes:
        agent (Agent): agent to execute the algorithm
        x0 (numpy.ndarray): initial condition
        x (numpy.ndarray): current value of the local solution
        shape (tuple): shape of the variable
        x_neigh (dict): dictionary containing the local solution of the (in-)neighbors
        enable_log (bool): True for enabling log
    """

    def __init__(self, agent: Agent, initial_condition: np.ndarray, enable_log=False):
        if not isinstance(agent, Agent):
            raise TypeError("Must provide an Agent as agent")

        super(SetMembership, self).__init__(agent, initial_condition, enable_log)

        self.parameter_set = None
        self.measurement_set = None
        self.measure_generator = None 
    
    def set_measure_generator(self, generator: Callable):
        """set the measure generator

        Args:
            generator: measure generator
        """
        self.measure_generator = generator

    def measure(self):
        """Takes a new measurement and updates parameter_set
        """
        new_set = self.measure_generator()
        self.measurement_set = deepcopy(new_set)

        if self.parameter_set is not None:
            if isinstance(self.parameter_set, AbstractSet):
                try:
                    self.parameter_set.intersection(new_set)
                except ValueError:
                    if isinstance(self.measurement_set, AbstractSet):
                        self.parameter_set = self.parameter_set.to_constraints() + new_set.to_constraints()
                    else:
                        if isinstance(self.measurement_set, Constraint):
                            self.parameter_set = self.parameter_set.to_constraints().append(new_set)
                        elif isinstance(self.measurement_set, list):
                            self.parameter_set = self.parameter_set.to_constraints().extend(new_set)
            elif isinstance(self.parameter_set, list):
                if isinstance(self.measurement_set, Constraint):
                    self.parameter_set = self.parameter_set.to_constraints().append(new_set)
                elif isinstance(self.measurement_set, list):
                    self.parameter_set = self.parameter_set.to_constraints().extend(new_set)
            elif isinstance(self.parameter_set, Constraint):
                if isinstance(self.measurement_set, Constraint):
                    self.parameter_set = [self.parameter_set].append(new_set)
                elif isinstance(self.measurement_set, list):
                    self.parameter_set = [self.parameter_set].extend(new_set)
        else:
            if isinstance(self.measurement_set, AbstractSet):
                self.parameter_set = new_set
            if isinstance(self.measurement_set, Constraint):
                self.parameter_set = [new_set]
            elif isinstance(self.measurement_set, list):
                self.parameter_set = new_set

    def _update_local_solution(self, x_new: np.ndarray):
        self.measure()
        if isinstance(self.parameter_set, AbstractSet):
            self.x = self.parameter_set.projection(x_new)
        else:  
            constraints = deepcopy(self.parameter_set)
            pb = ProjectionProblem(constraints, x_new)
            self.x = pb.solve()



class AsynchronousSetMembership(AsynchronousConsensus):
    """Asynchronous Distributed Set Membership Algorithm [FaGa19]_ 

    Args:
        agent (Agent): agent to execute the algorithm (must be a Agent)
        initial_condition (numpy.ndarray): initial condition
        enable_log (bool): True for enabling log

    Attributes:
        agent (Agent): agent to execute the algorithm
        x0 (numpy.ndarray): initial condition
        x (numpy.ndarray): current value of the local solution
        shape (tuple): shape of the variable
        x_neigh (dict): dictionary containing the local solution of the (in-)neighbors
        enable_log (bool): True for enabling log
        timestamp_sequence (list): list of timestamps
    """

    def __init__(self, agent: Agent, initial_condition: np.ndarray, **kwargs):
        if not isinstance(agent, Agent):
            raise TypeError("Must provide an Agent as agent")

        super(AsynchronousSetMembership, self).__init__(agent,
                                                        initial_condition, 
                                                        **kwargs)
        self.parameter_set = None
        self.measurement_set = None
        self.measure_generator = None 
    
    def set_measure_generator(self, generator: Callable):
        """set the measure generator

        Args:
            generator: measure generator
        """
        self.measure_generator = generator

    def measure(self):
        """Takes a new measurement and updates parameter_set
        """
        new_set = self.measure_generator()
        self.measurement_set = deepcopy(new_set)

        if self.parameter_set is not None:
            if isinstance(self.parameter_set, AbstractSet):
                try:
                    self.parameter_set.intersection(new_set)
                except ValueError:
                    if isinstance(self.measurement_set, AbstractSet):
                        self.parameter_set = self.parameter_set.to_constraints() + new_set.to_constraints()
                    else:
                        if isinstance(self.measurement_set, Constraint):
                            self.parameter_set = self.parameter_set.to_constraints().append(new_set)
                        elif isinstance(self.measurement_set, list):
                            self.parameter_set = self.parameter_set.to_constraints().extend(new_set)
            elif isinstance(self.parameter_set, list):
                if isinstance(self.measurement_set, Constraint):
                    self.parameter_set = self.parameter_set.to_constraints().append(new_set)
                elif isinstance(self.measurement_set, list):
                    self.parameter_set = self.parameter_set.to_constraints().extend(new_set)
            elif isinstance(self.parameter_set, Constraint):
                if isinstance(self.measurement_set, Constraint):
                    self.parameter_set = [self.parameter_set].append(new_set)
                elif isinstance(self.measurement_set, list):
                    self.parameter_set = [self.parameter_set].extend(new_set)
        else:
            if isinstance(self.measurement_set, AbstractSet):
                self.parameter_set = new_set
            if isinstance(self.measurement_set, Constraint):
                self.parameter_set = [new_set]
            elif isinstance(self.measurement_set, list):
                self.parameter_set = new_set

    def _update_local_solution(self, x_new: np.ndarray):
        self.measure()
        if isinstance(self.parameter_set, AbstractSet):
            self.x = self.parameter_set.projection(x_new)
        else:  
            constraints = deepcopy(self.parameter_set)
            pb = ProjectionProblem(constraints, x_new)
            self.x = pb.solve()

