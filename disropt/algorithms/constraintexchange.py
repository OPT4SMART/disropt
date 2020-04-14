import numpy as np
from ..agents import Agent
from .algorithm import Algorithm
from ..problems import Problem, LinearProblem
from ..functions import QuadraticForm, Variable
from itertools import combinations
from typing import Tuple

# TODO simplify code and add more detailed comments to methods
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
        x (numpy.ndarray): current value of the local solution
        B (numpy.ndarray): basis associated to the local solution
        shape (tuple): shape of the variable
        x_neigh (dict): dictionary containing the local solution of the (in-)neighbors
        sequence_x (numpy.ndarray): sequence of local solutions

    Args:
        agent (Agent): agent to execute the algorithm
        enable_log (bool): True to enable log
    """

    def __init__(self, agent: Agent, enable_log: bool = False):

        # initialize variables
        self.agent = agent
        self.enable_log = enable_log
        self.sequence_x = None
        self.shape = self.agent.problem.input_shape
        self.x_neigh = {}

        # initialize optimization variable and basis
        self.x = self.agent.problem.solve()
        self.B = self.compute_basis(self.agent.problem.constraints)

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
            if abs(val) < 1e-5:
                active_constr.append(con)

        # enumerating the possible combinations with d+1 constraints
        basis = active_constr
        for cand_basis_tuple in combinations(active_constr, np.max(self.shape) + 1):
            # convert candidate basis to list
            cand_basis = list(cand_basis_tuple)

            # solve problem with this candidate basis
            prob = Problem(self.agent.problem.objective_function, cand_basis)
            
            try:
                solution = prob.solve()
                if np.array_equal(solution, self.x):
                    # we have found a basis, break cycle
                    basis = cand_basis
                    break
            except:
                pass
        return basis

    def unique_constr(self, constraints):
        """Remove redundant constraints from given constraint list
        """

        # initialize shrunk list of constraints
        con_shrink = []

        # cycle over given constraints
        for con in constraints:
            if not con_shrink:
                con_shrink.append(con)
            else:
                check_equal = np.zeros((len(con_shrink), 1))
                # TODO: use numpy array_equal

                # cycle over already added constraints
                for idx, con_y in enumerate(con_shrink):
                    if self.compare_constr(con_y, con):
                        check_equal[idx] = 1
                n_zero = np.count_nonzero(check_equal)

                # add constraint if different from all the others
                if n_zero == 0:
                    con_shrink.append(con)
        return con_shrink

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

    def run(self, iterations=100, verbose: bool=False, **kwargs) -> np.ndarray:
        """Run the algorithm for a given number of iterations

        Args:
            iterations (int, optional): Number of iterations. Defaults to 100.
            verbose: If True print some information during the evolution of
                    the algorithm. Defaults to False.

        Returns:
            the sequence of computed solutions if enable_log=True.
        """
        if self.enable_log:
            dims = [iterations]
            for dim in self.shape:
                dims.append(dim)
            self.sequence_x = np.zeros(dims)

        for k in range(iterations):
            self.iterate_run()

            if self.enable_log:
                self.sequence_x[k] = self.x

            if verbose:
                if self.agent.id == 0:
                    print('Iteration {}'.format(k), end="\r")

        if self.enable_log:
            return self.sequence_x

    def get_result(self):
        """Return the current value of x

        Returns:
            numpy.ndarray: value of x
        """
        return self.x

class DistributedSimplex(Algorithm):
    """Distributed Simplex Algorithm [BuNo11]_

        This algorithm solves linear programs in standard form

        .. math::
            :nowrap:

            \\begin{split}
            \min_{x} \: & \: c^\\top x \\\\
            \mathrm{subj. to} \: & \: Ax = b \\\\
            & \: x \ge 0
            \\end{split}
        
        When reading the variable agent.problem.constraints, this class only
        considers equality constraints. Other constraints are discarded.

        .. warning::
            This class is currently under development

    Attributes:
        x (numpy.ndarray): current value of the complete solution
        J (float): current value of the cost
        x_basic (numpy.ndarray): current value of the basic solution
        B (numpy.ndarray): basis associated to the local solution
        n_constr (tuple): number of constraints of the problem
        B_neigh (dict): dictionary containing the local bases of (in-)neighbors
        A_init (numpy.ndarray): initial constraint matrix
        b_init (numpy.ndarray): initial constraint vector
        c_init (numpy.ndarray): initial cost vector
        sequence_x (numpy.ndarray): sequence of solutions
        sequence_J (numpy.ndarray): sequence of costs

    Args:
        agent (Agent): agent to execute the algorithm
        problem_size (list): total number of variables in the network. Defaults to None.
            If both problem_size and local_indices is provided, the complete solution
            vector will be computed.
        local_indices (list): indices of the agent's variables in the network, starting from 0. Defaults to None.
            If both problem_size and local_indices is provided, the complete solution
            vector will be computed.
        enable_log (bool): True to enable log
        stop_iterations (int): iterations with constant solution to stop algorithm. Defaults to None (disabled).
        big_M(float): cost of big-M variables. Defaults to 500.
    """

    def __init__(self, agent: Agent, problem_size: int = None, local_indices: list = None,
                enable_log: bool = False, stop_iterations: int = None, big_M: float = 500.0):

        # initialize class variables
        self.agent = agent
        self.problem_size = problem_size
        self.local_indices = local_indices
        self.enable_log = enable_log
        self.sequence_x = None
        self.sequence_J = None
        self.stop_iterations = stop_iterations
        self.big_M = big_M
        self.B_neigh = {}

        # simplex algorithm parameters
        self.par_iters = 100
        self.par_tol_leav = 1e-6
        self.par_tol_ent  = 1e-6

        # check if input data is correct
        if not isinstance(self.agent.problem, LinearProblem):
            raise TypeError("agent.problem must be an instance of LinearProblem")

        # other variables
        self.read_problem_data()
        self.n_constr = self.A_init.shape[0]
        self.x_basic = None # primal solution (basic)
        self.J = None # cost
        self.x_dual = None # dual solution
        self.x = None # primal solution (full)

        self.save_x_dual = True # true if self.x_dual can be populated
        self.save_x = False # true if self.x can be populated
        self.indices_available = False # true if column indices are available

        # check index consistency and, in case, adjust flags
        if self.problem_size is not None and self.local_indices is not None:
            self.check_index_consistency()
            self.save_x = True
            self.indices_available = True
        
        # initialize algorithm
        self.initialize()
    
    def check_index_consistency(self):
        """Check consistency of local indices, problem_size and constraint matrix
        """
        if not isinstance(self.local_indices, list):
            raise TypeError("local_indices must be a list")
        if len(self.local_indices) != self.A_init.shape[1]:
            raise ValueError("The size of local_indices must equal the number of variables in agent.problem")
        if self.problem_size < self.A_init.shape[1]:
            raise ValueError("problem_size must be greater than or equal to the size of local_indices")
    
    def read_problem_data(self):
        """Read local problem data from agent.problem. The data is
        saved in order to be solved as a standard form problem.
        """
        A_list = []
        b_list = []

        # loop through equality constraints
        for constraint in self.agent.problem.constraints:
            if constraint.is_equality:
                A_c, b_c = constraint.get_parameters()

                # constraint is written in the form A^\top x + b = 0
                A_list.append(A_c.transpose())
                b_list.append(-b_c.flatten()) # save as 1D vector
        
        # build initial constraint matrix and vector from lists
        self.b_init = np.hstack(tuple(b_list))[:, None] # save as 2D vertical vector
        self.A_init = np.vstack(tuple(A_list))

        # read initial cost vector
        cost_parameters = self.agent.problem.objective_function.get_parameters()
        self.c_init = cost_parameters[0] # save as 2D vertical vector

    def initialize(self):
        """Evaluate a first solution and basis starting from
        agent's constraints through the Big-M method
        """

        # change sign of constraint rows where b is negative
        for j in range(self.b_init.shape[0]):
            if self.b_init[j] < 0:
                self.b_init[j] *= -1
                self.A_init[j, :] *= -1
        
        # tableau for agent's initial matrices
        H_init = np.vstack((self.c_init.transpose(), self.A_init))

        # tableau for big-M (initial lex-feasible basis)
        A_bigM = np.eye(self.n_constr)
        c_bigM = self.big_M*np.ones((1, self.n_constr))
        H_bigM = np.vstack((c_bigM, A_bigM))

        # add column partitioning information if available
        if self.indices_available:
            # initial columns are assigned local_indices
            H_init = np.vstack((H_init, np.array(self.local_indices)))

            # big-M columns are assigned indices after problem_size
            indices_bigM = np.arange(self.problem_size, self.problem_size + self.n_constr)
            H_bigM = np.vstack((H_bigM, np.array(indices_bigM)))

        # prepare lex-sorted tableau
        H_sort, basis_idx = sort_tableau(H_init, H_bigM)

        if self.indices_available:
            tableau = H_sort[0:-1, :]
        else:
            tableau = H_sort

        # run simplex method
        sol = lex_simplex(tableau, self.b_init, basis_idx, self.par_iters, self.par_tol_ent, self.par_tol_leav)

        # update local solution
        self._update_local_solution(H_sort[:, sol[1]])

    def _update_local_solution(self, basis):
        """Save solution data using current basis information
        """
        # extract constraint matrix and cost vector
        if self.indices_available:
            A = basis[1:-1, :]
        else:
            A = basis[1:, :]
        c = basis[0, :]

        # compute basic variables
        x_basic = np.linalg.solve(A, self.b_init)

        # compute dual solution
        y = np.linalg.solve(A.transpose(), c)

        # save
        self.B = basis
        self.x_basic = x_basic
        self.x_dual = y
        self.J = c @ x_basic

        # compute entire vector if column partitioning information is available
        if self.indices_available:
            self.x = np.zeros((self.problem_size + self.n_constr, 1))
            basis_idx = basis[-1, :].astype(int)
            self.x[basis_idx, :] = self.x_basic
    
    def get_basis(self):
        """Return current basis
        """
        if self.indices_available:
            return self.B[0:-1]
        else:
            return self.B
    
    def get_result(self):
        """Return the current value of the solution
    
        Returns:
            tuple of nd.ndarray (primal, primal_basic, dual, cost): value of primal solution, primal basic solution, dual solution, cost
        """
        return (self.x, self.x_basic, self.x_dual, self.J)

    def iterate_run(self):
        """Run a single iterate of the algorithm
        """

        # exchange columns with neighbors - TODO implement exchange of "null" symbol
        data = self.agent.neighbors_exchange(self.B)

        # concatenate all the received bases
        for neigh in data:
            self.B_neigh[neigh] = data[neigh]
        H_list = list(self.B_neigh.values())
        
        # add tableau for agent's initial matrices
        if self.indices_available:
            H_list.append(np.vstack((self.c_init.transpose(), self.A_init, self.local_indices)))
        else:
            H_list.append(np.vstack((self.c_init.transpose(), self.A_init)))

        # prepare lex-sorted tableau
        H_full = np.hstack(tuple(H_list))
        H_sort, basis_idx = sort_tableau(H_full, self.B)

        if self.indices_available:
            tableau = H_sort[0:-1, :]
        else:
            tableau = H_sort

        # run simplex method
        sol = lex_simplex(tableau, self.b_init, basis_idx, self.par_iters, self.par_tol_ent, self.par_tol_leav)

        # update local solution
        self._update_local_solution(H_sort[:, sol[1]])

    def run(self, iterations=100, verbose: bool=False, **kwargs) -> np.ndarray:
        """Run the algorithm for a given number of iterations

        Args:
            iterations (int, optional): Maximum number of iterations. Defaults to 100.
            verbose: If True print some information during the evolution of the algorithm. Defaults to False.

        Raises:
            TypeError: The number of iterations must be an int
        
        Returns:
            return a tuple (x, J) with the sequence of solutions and costs if enable_log=True.
        """
        if not isinstance(iterations, int):
            raise TypeError("The number of iterations must be an int")

        if self.enable_log:
            # initialize cost sequence
            dims = [iterations]
            self.sequence_J = np.zeros(dims)

            # initialize solution sequence
            if self.save_x:
                dims.extend(self.x.shape)
                self.sequence_x = np.zeros(dims)
        
        # initialize counter for stopping criterion
        counter = 0
        last_iter = np.copy(iterations)

        for k in range(iterations):
            # store previous basis
            prev_B = self.B

            # perform an iteration
            self.iterate_run()

            # store solution and cost sequence
            if self.enable_log:
                self.sequence_J[k] = self.J

                if self.save_x:
                    self.sequence_x[k] = self.x
            
            # print information
            if verbose and self.agent.id == 0:
                print('Iteration {}'.format(k), end="\r")

            # increase counter if solution has not changed, otherwise reset
            if np.linalg.norm(prev_B - self.B) < 1e-6:
                counter += 1
            else:
                counter = 0
            
            # check termination condition
            if self.stop_iterations is not None and counter > self.stop_iterations:
                # convergence detected
                self.agent.neighbors_send(self.B) # broadcast local basis a last time
                last_iter = k+1
                break
        
                # TODO check if problem is infeasible (i.e. if there are still big-M columns in the basis)

        # return sequences
        if self.enable_log:
            if self.save_x:
                return (self.sequence_x.take(np.arange(0,last_iter), axis=0), self.sequence_J[0:last_iter])
            else:
                return (None, self.sequence_J[0:last_iter])


class DualDistributedSimplex(DistributedSimplex):
    """Distributed Simplex Algorithm on dual problem [BuNo11]_

        This algorithm solves linear programs of the form

        .. math::
            :nowrap:

            \\begin{split}
            \max_{x} \: & \: c^\\top x \\\\
            \mathrm{subj. to} \: & \: Ax \le b
            \\end{split}
        
        This class runs the Distributed Simplex algorithm on the (standard form) dual problem.

        .. warning::
            This class is currently under development

    Args:
        agent (Agent): agent to execute the algorithm
        num_constraints (list): total number of constraints in the network. Defaults to None.
            If both num_constraints and local_indices is provided, the complete dual solution
            vector will be computed.
        local_indices (list): indices of the agent's constraints in the network, starting from 0. Defaults to None.
            If both num_constraints and local_indices is provided, the complete dual solution
            vector will be computed.
        enable_log (bool): True to enable log
        stop_iterations (int): iterations with constant solution to stop algorithm. Defaults to None (disabled).
        big_M(float): cost of big-M variables. Defaults to 500.
    """

    def __init__(self, agent: Agent, num_constraints: int = None, local_indices: list = None,
                enable_log: bool = False, stop_iterations: int = None, big_M: float = 500.0):
        # call init of parent class
        super().__init__(agent=agent, problem_size=num_constraints, local_indices=local_indices,
                enable_log=enable_log, stop_iterations=stop_iterations, big_M=big_M)
        
        # swap values of save_x and save_x_dual
        self.save_x, self.save_x_dual = self.save_x_dual, self.save_x

    def check_index_consistency(self):
        """Check consistency of local indices, num_constraints and constraint matrix
        """
        # re-adapted error messages for the dual case
        if not isinstance(self.local_indices, list):
            raise TypeError("local_indices must be a list")
        if len(self.local_indices) != self.A_init.shape[1]:
            raise ValueError("The size of local_indices must equal the number of constraints in agent.problem")
        if self.problem_size < self.A_init.shape[1]:
            raise ValueError("num_constraints must be greater than or equal to the size of local_indices")

    def read_problem_data(self):
        """Read local problem data from agent.problem. The data is
        saved in order to be solved as a standard form problem.
        """
        A_list = []
        b_list = []

        # loop through inequality constraints
        for constraint in self.agent.problem.constraints:
            if constraint.is_inequality:
                A_c, b_c = constraint.get_parameters()

                # constraint is written in the form A^\top x + b <= 0
                A_list.append(A_c)
                b_list.append(-b_c.flatten()) # save as 1D vector
        
        # build initial constraint matrix and cost vector from lists
        self.c_init = np.hstack(tuple(b_list))[:, None] # save as 2D vertical vector
        self.A_init = np.vstack(tuple(A_list))

        # read initial constraint vector
        cost_parameters = self.agent.problem.objective_function.get_parameters()
        self.b_init = -cost_parameters[0] # save as 2D vertical vector (minus sign for maximization)

    def _update_local_solution(self, basis):
        """Save solution data using current basis information
        """
        # call method of parent class
        super()._update_local_solution(basis)

        # swap primal and dual solutions
        self.x, self.x_dual = self.x_dual, self.x
    
    def get_result(self):
        """Return the current value of the solution
    
        Returns:
            tuple of nd.ndarray (primal, dual_basic, dual, cost): value of primal solution, dual basic solution, dual solution, cost
        """
        return (self.x, self.x_basic, self.x_dual, self.J)

#####################################
# Utilities

def lexnonpositive(vector: np.ndarray, tol: float) -> bool:
    """tests if a vector is lexicographically non-positive, i.e.,
    it is a vector made of zeros or its first non-zero component is negative

    Args:
        vector (np.ndarray): input vector
        tol (float): numerical tolerance

    Returns:
        is_nonpositive (bool): True if vector is non-positive
    """

    for elem in vector:
        if abs(elem) < tol:  # Treat it as a zero
            continue
        # if the component is positive return false
        if elem > 0:
            return False

        # else return true
        break

    return True


def enteringcolumn(tableau: np.ndarray, ind_basic: np.ndarray, tol: float) -> int:
    """Lex-simplex routine that selects an entering column
    as described in Jones et al (Automatica 2007)

    Args:
        tableau (np.ndarray): lexicographically sorted tableau
        ind_basic (np.ndarray): indices of current lex-feasible basis
        tol (float): numerical tolerance

    Returns:
        index_e (int): index of entering column (or None if basis is already optimal)
    """

    ncols = tableau.shape[1]

    # extract the constraint matrix and the basic columns
    A = tableau[1:, :]
    basis = A[:, ind_basic]

    # evaluate reduced costs and construct matrix [kappa@c kappa] for test
    kappa = np.eye(ncols)
    kappa[:, ind_basic] -= np.linalg.solve(basis, A).transpose()
    cost = tableau[0, :]
    reduced_costs = kappa @ cost
    kc_k_matrix = np.column_stack((reduced_costs, kappa))

    # cycle on non-basic columns
    for ii in range(ncols):
        if ii not in ind_basic:
            test_vector = kc_k_matrix[ii, :]

            if lexnonpositive(test_vector, tol):
                return ii

    return None


def leavingcolumn(index_e: int, b: np.ndarray, tableau: np.ndarray,
    tol: float, ind_basic: np.ndarray) -> int:
    """Lex-simplex routine that selects a leaving column
    with the lex-ratio test described in Jones et al (Automatica 2007)

    Args:
        index_e (int): current entering column
        b (np.ndarray): right-hand side vector of constraints
        tableau (np.ndarray): lexicographically sorted tableau
        tol (float): numerical tolerance
        ind_basic (np.ndarray): indices of current lex-feasible basis

    Returns:
        index_l (int): index of leaving column (or None if problem is unbounded)
    """

    # Extract basic columns from tableau
    basis = tableau[1:, ind_basic]

    # construct beta*A_e (with beta = B^-1)
    A_e = tableau[1:, index_e][:, None]
    beta_A = np.linalg.solve(basis, A_e)

    # find positive entries of beta_A
    ind_pos = np.flatnonzero(beta_A > tol)
    if ind_pos.size == 0:
        return None # problem is unbounded
    
    # construct [beta*b beta] matrix
    beta = np.linalg.inv(basis)
    beta_b = np.linalg.solve(basis, b)
    M = np.column_stack((beta_b, beta))

    # compute ratios
    ratios = M[ind_pos, :] / beta_A[ind_pos]

    # sort rows lexicographically
    ind_sort = np.lexsort(np.rot90(ratios))

    # return lex-minimal row index
    return ind_pos[ind_sort[0]]


def lex_simplex(tableau: np.ndarray, b: np.ndarray, ind_basic: np.ndarray,
    max_iters: int, tol_ent: float, tol_leav: float) -> Tuple[int, np.ndarray, int]:
    """Lexicographic simplex algorithm
    Args:
        tableau (np.ndarray): lexicographically sorted tableau
        b (np.ndarray): right-hand side vector of constraints
        ind_basic (np.ndarray): indices of current lex-feasible basis
        max_iters (int): maximum number of iterations
        tol_ent (float): numerical tolerance to evaluate entering column
        tol_leav (float): numerical tolerance to evaluate leaving column

    Returns:
        unbounded (int): True if problem is unbounded, False otherwise
        new_ind_basic (np.ndarray): new basic variables
        nn (int): number of performed iterations
    """

    new_ind_basic = np.copy(ind_basic)
    
    for nn in range(max_iters):
        # check for an entering column
        index_EC = enteringcolumn(tableau, new_ind_basic, tol_ent)

        if index_EC == None:  # Problem solved
            return False, new_ind_basic, nn+1

        # check for a leaving column
        index_LC = leavingcolumn(index_EC, b, tableau, tol_leav, new_ind_basic)

        if index_LC == None:  # unbounded case
            return True, None, nn+1

        # pivoting
        new_ind_basic[index_LC] = index_EC

    return False, new_ind_basic, nn+1


def sort_tableau(H_data: np.ndarray, B: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Lexicographically sort all the columns and remove duplicates
    Args:
        H_data (np.ndarray): tableau without basic columns
        B (np.ndarray): basic columns

    Returns:
        H_sort (np.ndarray): lexicographically sorted tableau without duplicates
        ind_basic_sort (np.ndarray): indices of basic columns
    """
    # make tableau with B at the beginning
    H = np.hstack((B, H_data))
    ind_basic = np.arange(0, B.shape[1]) # basic indices: 0, ..., B.shape[1] - 1

    # remove duplicate columns - TODO use a tolerance
    H_unique, ind_unique = np.unique(H, return_inverse=True, axis=1)

    # lexicographically sort tableau
    ind_sort = np.lexsort(np.flip(H_unique, 0)).tolist()
    H_sort = H_unique[:, ind_sort]

    # compute new position of basic columns
    ind_basic_sort = [ind_sort.index(i) for i in ind_unique[ind_basic]]

    return H_sort, np.array(ind_basic_sort).astype(int)
