import numpy as np
from ..agents import Agent
from .algorithm import Algorithm
from ..problems import Problem
from ..functions import QuadraticForm, Variable
from itertools import combinations

class ConstraintsConsensus(Algorithm):
    """Constraints Consensus Algorithm [NoBu11]_
    
    This algorithm solves convex and abstract programs in the form

        .. math::
            :nowrap:

            \\begin{split}
            \min_{x} \: & \: c^\\top x \\\\
            \mathrm{subj. to} \: & \: x\\in \\bigcap_{i=1}^{N} X_i \\\\
            & \: x \ge 0
            \\end{split}

    Attributes:
        agent (Agent): agent to execute the algorithm
        x (numpy.ndarray): current value of the local solution
        B (numpy.ndarray): basis associated to the local solution
        shape (tuple): shape of the variable
        x_neigh (dict): dictionary containing the local solution of the (in-)neighbors
        enable_log (bool): True for enabling log
        stopping_criterion(bool): True for imposing a stopping criterion
        stopping_iteration (int): Iterations for stopping criterion

    Args:
        agent (Agent): agent to execute the algorithm
        enable_log (bool): True for enabling log
        stopping_criterion(bool): True for imposing a stopping criterion
        stopping_iteration (int): Iterations for stopping criterion
    """

    def __init__(self, agent: Agent, enable_log: bool = False,
                 stopping_criterion: bool = False, stopping_iteration: int = 0):
        super(ConstraintsConsensus, self).__init__(agent, enable_log)
        if not isinstance(stopping_criterion, bool):
            raise ValueError("stopping_criterion must be a bool")
        if not isinstance(stopping_iteration, int):
            raise ValueError("stopping_iteration must be an int")

        # initialize variables
        self.agent = agent
        self.enable_log = enable_log
        self.sequence = None
        self.stopping_criterion = stopping_criterion
        self.stopping_iteration = stopping_iteration
        self.shape = self.agent.problem.input_shape
        self.x_neigh = {}

        # initialize optimization variable and basis
        # try:
        self.x = self.agent.problem.solve()
        self.B = self.compute_basis(self.agent.problem.constraints)
        # except:
        #     print(self.x)
        #     exit()

    def get_basis(self):
        """Return agent basis
        """
        return self.B

    def compute_basis(self, constraints):
        """Compute a (minimal) basis for the given constraint list
        """

        # remove redundant constraints
        constraints_unique = self.unique_constr(constraints)

        # find active constraints
        active_constr = []
        for con in constraints_unique:
            # try:
            val = con.function.eval(self.x)
            # except:
            #     print(self.x)
            if abs(val) < 1e-5:
                active_constr.append(con)

        # enumerating the possible combinations with d+1 constraints
        basis = []
        basis = active_constr
        for indices in combinations(active_constr, np.max(self.shape) + 1):
            # extract candidate basis
            cand_basis = list(indices)

            # solve problem with this candidate basis
            prob = Problem(self.agent.problem.objective_function, cand_basis)
            
            try:
                solution = prob.solve()
                if np.array_equal(solution, self.x):
                    # we have found a basis, break cycle
                    basis = indices
                    break
            except:
                pass
        return basis

    def unique_constr(self, constraints):
        """Remove redundant constraints from given constraint list
        """

        # initialize shrunk constraint list
        con_shrunk = []

        # cycle over given constraints
        for con in constraints:
            if not con_shrunk:
                con_shrunk.append(con)
            else:
                check_equal = np.zeros((len(con_shrunk), 1))
                # TODO: use np array_equal

                # cycle over already added constraints
                for idx, con_y in enumerate(con_shrunk):
                    if self.compare_constr(con_y, con):
                        check_equal[idx] = 1
                n_zero = np.count_nonzero(check_equal)

                # add constraint if different from all the others
                if n_zero == 0:
                    con_shrunk.append(con)
        return con_shrunk

    def compare_constr(self, a, b):
        """Compare two constraints to check whether they are equal
        """

        # test for affine constraints
        if (a.is_affine and b.is_affine):
            A_a, b_a = a.get_parameters()
            A_b, b_b = b.get_parameters()
            if np.array_equal(A_a, A_b) and np.array_equal(b_a, b_b):
                return True
            else:
                return False
        
        # test for quadratic constraints
        elif (a.is_quadratic and b.is_quadratic):
            P_a, q_a, r_a = a.get_parameters()
            P_b, q_b, r_b = b.get_parameters()
            if np.array_equal(P_a, P_b) and np.array_equal(q_a, q_b) and np.array_equal(r_a, r_b):
                return True
            else:
                return False
        else:
            return False

    def constr_to_dict(self, constraints):
        """Convert constraint list to dictionary
        """

        con_dict = {}
        for key, con in enumerate(constraints):
            if con.is_affine:
                A, b = con.get_parameters()
                con_dict[key] = {"kind": "affine", "sign": con.sign, "Amat": A,
                                 "bmat": b}
            if con.is_quadratic:
                P, q, r = con.get_parameters()
                con_dict[key] = {"kind": "quadratic", "sign": con.sign, "Pmat": P,
                                 "qmat": q, "rmat": r}
        return con_dict

    def dict_to_constr(self, dictio):
        """Convert dictionary to constraint list
        """

        dict_len = len(dictio)
        constr = []
        x = Variable(int(np.max(self.shape)))
        for idx in np.arange(dict_len):
            con_dict = dictio.get(idx)
            str_kind = con_dict.get("kind")
            str_sign = con_dict.get("sign")
            if str_kind == "quadratic":
                P = con_dict.get("Amat")
                q = con_dict.get("qmat")
                r = con_dict.get("rmat")
                g = QuadraticForm(x, P, q, r)
                if str_sign == "==":
                    con = g == 0
                else:
                    con = g <= 0
            elif str_kind == "affine":
                A = con_dict.get("Amat")
                b = con_dict.get("bmat")
                if str_sign == "==":
                    con = A @ x + b == 0
                else:
                    con = A @ x + b <= 0
            else:
                raise ValueError(
                    "Current supported constraints are affine and quadratic")
            constr.append(con)
        return constr

    def iterate_run(self):
        """Run a single iterate of the algorithm
        """

        # convert basis to dictionary and send to neighbors
        basis_dict = self.constr_to_dict(self.B)
        data = self.agent.neighbors_exchange(basis_dict)

        # create list of agent's constraints + received constraints
        constraints = []
        constraints.extend(self.agent.problem.constraints)
        for neigh in data:
            constraints.extend(self.dict_to_constr(data[neigh]))

        # solve problem
        problem = Problem(self.agent.problem.objective_function, constraints)
        self.x = problem.solve()

        # compute new basis
        self.B = self.compute_basis(constraints)

    def run(self, iterations=100, verbose: bool = False):
        """Run the algorithm for a given number of iterations

        Args:
            iterations (int, optional): Number of iterations. Defaults to 100.
            verbose: If True print some information during the evolution of
                    the algorithm. Defaults to False.
        """
        if self.enable_log:
            dims = [iterations]
            for dim in self.shape:
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
        """Return the current value of x

        Returns:
            numpy.ndarray: value of x
        """
        return self.x
