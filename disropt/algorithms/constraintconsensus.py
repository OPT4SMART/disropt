import numpy as np
import time
from ..agents import Agent
from .algorithm import Algorithm
from ..utils.LexLPUtils import *
from ..problems import Problem
from ..problems import LinearProblem


class ConstraintConsensus(Algorithm):
    """Constraint Consensus Algorithm [NoBu11]_ (children of Algorithm)
        for convex and abstract programs in the form

        .. math::
            :nowrap:

            \\begin{split}
            \min_{x} \hspace{1.1cm} & \: c^\\top x \\\\
            \mathrm{subj. to} \: x\\in \\cap_{i=1}^{N} X_i \\\\
            & \: x \ge 0
            \\end{split}
        
        .. warning::
            This class is currently under development

    Attributes:
        agent (AwareAgent): agent to execute the algorithm
        x (numpy.ndarray): current value of the local solution
        B (numpy.ndarray): basis associated to the local solution
        shape (tuple): shape of the variable
        x_neigh (dict): dictionary containing the local solution of
                         the (in-)neighbors
        enable_log (bool): True for enabling log
        stopping_criterion(bool): True for imposing a stopping criterion
        stopping_iteration (int): Iterations for stopping criterion

    Args:
        agent (Agent): agent to execute the algorithm
        initial_condition (numpy.ndarray): initial condition
        enable_log (bool): True for enabling log
        stopping_criterion(bool): True for imposing a stopping criterion
        stopping_iteration (int): Iterations for stopping criterion
    """

    def __init__(self, agent: Agent, enable_log: bool=False,
                 stopping_criterion: bool=False, stopping_iteration: int = 0):
        super(ConstraintConsensus, self).__init__(agent, enable_log)
        if not isinstance(stopping_criterion, bool):
            raise ValueError("stopping_criterion must be a bool")
        if not isinstance(stopping_iteration, int):
            raise ValueError("stopping_iteration must be an int")

        self.agent = agent
        self.enable_log = enable_log
        self.sequence = None
        self.stopping_criterion = stopping_criterion
        self.stopping_iteration = stopping_iteration

        self.x = self.agent.problem.solve()
        self.evaluate_basis(self, self.agent.problem.constraints)

        self.shape = self.x.shape

        self.x_neigh = {}

    def get_basis(self):
        """Return agent basis

        """
        return self.B

    def evaluate_basis(self, constraints):
        cand_basis = []
        for con in constraints:
            val = constraints.function.eval(self.x)
            if abs(val) < 1e-6:
                cand_basis.append(con)
        cand_basis = self.unique_constr(cand_basis)
        basis = []
        for idx, con in enumerate(cand_basis):
            tmp_basis = cand_basis.pop(idx)
            p_tmp = Problem(self.ag.problem.objective_function,
                                    tmp_basis)
            sol_tmp = p_tmp.solve()
            if np.linalg.norm(sol_tmp-self.x) >= 1e-6:
                basis.append(con)
        self.B = basis

    def unique_constr(x):
        y = x
        for i, a in enumerate(x):
            if any(self.compare_constr(a, b) for b in x[:i]):
                y.pop(i)
        return y

    def compare_constr(a, b):
        if (a.is_affine and b.is_affine) or ((a.is_quadratic and b.is_quadratic)):
            A_a, b_a = a.get_parameters()
            A_b, b_b = b.get_parameters()
            if (np.linalg.norm(A_a-A_b) <= 1e-6) and (np.linalg.norm(b_a-b_b) <= 1e-6):
                return True
            else:
                return False
        else:
            return False

    def get_result(self):
        """Return the value of the solution
        """
        return self.x

    def iterate_run(self):
        """Run a single iterate of the algorithm
        """
        data = self.agent.neighbors_exchange(self.B)
        constraints = []
        constraints.append(self.agent.problem.constraints)
        for neigh in data:
            constraints.append(data[neigh])
        p_tmp = Problem(self.ag.problem.objective_function, tmp_basis)
        self.x = p_tmp.solve()
        self.evaluate_basis(self, constraints)

    def run(self, iterations=100, verbose: bool=False):
        """Run the algorithm for a given number of iterations

        Args:
            iterations (int, optional): Number of iterations. Defaults to 100.
            verbose: If True print some information during the evolution of
                    the algorithm. Defaults to False.
        """
        if self.enable_log:
            dims = [iterations]
            for dim in self.x.shape:
                dims.append(dim)
            self.sequence = np.zeros(dims)

        for k in range(iterations):
            self.iterate_run()

            if self.enable_log:
                self.sequence[k] = self.x

            if verbose:
                if self.agent.id == 0:
                    print('Iteration {}'.format(k), end="\r")

        if self.enable_log:
            return self.sequence

    def get_result(self):
        """Return the actual value of x

        Returns:
            numpy.ndarray: value of x
        """
        return self.x


class LinearConstraintConsensus(ConstraintConsensus):
    """Linear Constraint Consensus by Distributed Simplex Algorithm

        This works for linear programs in the form
        .. math::
            :nowrap:

            \\begin{split}
            \min_{x} \hspace{1.1cm} & \: c^\\top x \\\\
            \mathrm{subj. to} \: Ax + b = 0 \\\\
            & \: x \ge 0
            \\end{split}


        Or linear programs in the form

        .. math::
            :nowrap:

            \\begin{split}
            \min_{x} \hspace{1.1cm} & \: c^\\top x \\\\
            \mathrm{subj. to} \: Ax + b \le 0
            \\end{split}
        
        .. warning::
            This class is currently under development


    Attributes:
        agent (AwareAgent): agent to execute the algorithm
        x (numpy.ndarray): current value of the local solution
        B (numpy.ndarray): basis associated to the local solution
        shape (tuple): shape of the variable
        x_neigh (dict): dictionary containing the local solution of
        the (in-)neighbors in terms of basis
        enable_log (bool): True for enabling log
        A_init (numpy.ndarray): initial constraint matrix
        b_init (numpy.ndarray): initial constraint rhs
        c_init (numpy.ndarray): initial cost function

    Args:
        agent (Agent): agent to execute the algorithm
        initial_condition (numpy.ndarray): initial condition
        enable_log (bool): True for enabling log
    """

    def __init__(self, agent: Agent, enable_log: bool=False,
                 stopping_criterion: bool=False, stopping_iteration: int = 0,
                 is_standardform: bool=True, big_M: float=500.0):
        super(LinearConstraintConsensus, self).__init__(agent, enable_log,
                                                        stopping_criterion,
                                                        stopping_iteration)

        if not isinstance(self.agent.problem, LinearProblem):
            raise ValueError(
                    "agent.problem must be an instance of LinearProblem")

        if not isinstance(is_standardform, bool):
            raise ValueError(
                    "is_standardform must be bool")

        self.is_standardform = is_standardform
        self.big_M = big_M
        self.initialize()

        self.shape = self.x.shape

        self.x_neigh = {}

    def initialize(self):
        """Evaluate a first solution and basis starting from
        agent's constraints through the Big-M method
        """
        constr_list = self.agent.problem.constraints
        b_tmp = np.ndarray(0)
        onfirst = True
        for idx, constraint in enumerate(constr_list):
            A_c, b_c = constraint.get_parameters()
            A = A_c.transpose()
            if self.is_standardform:
                if onfirst:
                    A_tmp = np.ndarray((A.shape[0], 0))
                    onfirst = False
                A_tmp = np.hstack((A_tmp, A))
            else:
                if onfirst:
                    A_tmp = np.ndarray((0, A.shape[1]))
                    onfirst = False
                A_tmp = np.vstack((A_tmp, A))
            b_tmp = np.hstack((b_tmp, -b_c.flatten()))
        pars = self.agent.problem.objective_function.get_parameters()
        c_tmp = pars[0]
        c_tmp = c_tmp.transpose()
        self.A_init = A_tmp
        self.b_init = b_tmp
        self.c_init = c_tmp
        if self.is_standardform:
            shape = self.A_init.shape[0]
        else:
            shape = self.A_init.shape[1]
        BigM_H = create_bigM(shape, self.big_M,
                             self.is_standardform)
        if self.is_standardform:
            H_data = np.vstack((c_tmp, A_tmp))
            b = self.b_init
        else:
            H_data = tostandard(A_tmp, b_tmp)
            b = self.c_init

        init_data = init_lexsimplex(H_data, BigM_H, shape)
        H_sort = init_data[0]
        sol = simplex(H_sort, b, init_data[1], 100, 1e-6, 1e-6)

        sol_data = retrievelexsol(H_sort[:, sol[1]], b,
                                  self.is_standardform, shape)
        self.B = sol_data[0]
        self.x = sol_data[1]

    def get_basis(self):
        """Evaluate solution from neighbors and local data

        """
        return self.B

    def iterate_run(self):
        """Run a single iterate of the algorithm
        """
        data = self.agent.neighbors_exchange(self.B)
        if self.is_standardform:
            H_data = np.vstack((self.c_init, self.A_init))
            BigM_H = create_bigM(np.max(self.shape), self.big_M,
                                 self.is_standardform)
            H_data = np.hstack((H_data, BigM_H))
            b = self.b_init
        else:
            H_data = np.hstack((self.A_init, self.b_init))
            BigM_H = create_bigM(np.max(self.shape), self.big_M,
                                 self.is_standardform)
            H_data = np.vstack((H_data, BigM_H))
            b = self.c_init

        for neigh in data:
            self.x_neigh[neigh] = data[neigh]
            if self.is_standardform:
                H_data = np.hstack((H_data, self.x_neigh[neigh]))
            else:
                H_data = np.vstack((H_data, self.x_neigh[neigh]))

        init_data = init_lexsimplex(H_data, self.B, self.shape)
        sol = simplex(init_data[0], b, init_data[1], 100, 1e-6, 1e-6)
        H_sort = init_data[0]
        sol_data = retrievelexsol(H_sort[:, sol[1]], b,
                                  self.is_standardform, self.shape)
        self.B = sol_data[0]
        self.x = sol_data[1]

    def run(self, iterations=100):
        """Run the algorithm for a given number of iterations

        Args:
            iterations (int, optional): Number of iterations. Defaults to 100.
        """
        if self.enable_log and not self.stopping_criterion:
            dims = [iterations]
            for dim in self.x.shape:
                dims.append(dim)
            self.sequence = np.zeros(dims)
        if self.enable_log and self.stopping_criterion:
            dims = [0]
            for dim in self.x.shape:
                dims.append(dim)
            self.sequence = np.zeros(dims)

        if self.stopping_criterion:
            stop = False
            counter = 0
            while ~stop:
                prev_x = self.x
                self.iterate_run()
                if np.linalg.norm(prev_x-self.x) < 1e-6:
                    counter += 1
                if counter > self.stopping_iteration:
                    stop = True
                if self.enable_log:
                    np.append(self.sequence, self.x[:, :, None], axis=0)
        else:
            for k in range(iterations):
                self.iterate_run()
                if self.enable_log:
                    self.sequence[k] = self.x

        if self.enable_log:
            return self.sequence
