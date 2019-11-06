import numpy as np
import time
from ..agents import Agent
from .algorithm import Algorithm
from ..problems import Problem
from ..problems import LinearProblem
from ..functions import QuadraticForm, Variable
from itertools import combinations
import copy


class ConstraintConsensus(Algorithm):
    """Constraint Consensus Algorithm [NoBu11]_
        for convex and abstract programs in the form

        .. math::
            :nowrap:

            \\begin{split}
            \min_{x} \hspace{1.1cm} & \: c^\\top x \\\\
            \mathrm{subj. to} &\: x\\in \\cap_{i=1}^{N} X_i \\\\
            & \: x \ge 0
            \\end{split}

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

    def __init__(self, agent: Agent, enable_log: bool = False,
                 stopping_criterion: bool = False, stopping_iteration: int = 0):
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
        self.shape = self.agent.problem.input_shape

        # try:
        self.x = self.agent.problem.solve()
#            self.B = self.evaluate_basis(self.agent.problem.constraints)
        # except:
        #     print(self.x)
        #     exit()

        # print("agent {} no error".format(self.agent.id))
        # exit()
        self.B = self.evaluate_basis(self.agent.problem.constraints)

        self.x_neigh = {}

    def get_basis(self):
        """Return agent basis

        """
        return self.B

    def evaluate_basis(self, constraints):
        constraints_unique = self.unique_constr(constraints)
        cand_basis = []
        for con in constraints_unique:
            try:
                val = con.function.eval(self.x)
            except:
                print(self.x)
            if abs(val) < 1e-5:
                cand_basis.append(con)
        basis = []
        f_star = self.agent.problem.objective_function.eval(self.x)
        basis = cand_basis
        for indices in combinations(cand_basis, np.max(self.shape) + 1):
            tmp_basis = list(indices)
            p_tmp = Problem(self.agent.problem.objective_function, tmp_basis)
            try:
                sol_tmp = p_tmp.solve()
                if np.array_equal(sol_tmp, self.x):
                    basis = indices
                    break
            except:
                pass
        return basis

    def unique_constr(self, x):
        y = []
        for con in x:
            if not y:
                y.append(con)
            else:
                check_equal = np.zeros((len(y), 1))
                # TODO: use np array_equal
                for idx, con_y in enumerate(y):
                    if self.compare_constr(con_y, con):
                        check_equal[idx] = 1
                n_zero = np.count_nonzero(check_equal)
                if n_zero == 0:
                    y.append(con)
        return y

    def compare_constr(self, a, b):
        if (a.is_affine and b.is_affine):
            A_a, b_a = a.get_parameters()
            A_b, b_b = b.get_parameters()
            if np.array_equal(A_a, A_b) and np.array_equal(b_a, b_b):
                return True
            else:
                return False
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

    def get_result(self):
        """Return the value of the solution
        """
        return self.x

    def iterate_run(self):
        """Run a single iterate of the algorithm
        """
        basis_dict = self.constr_to_dict(self.B)
        data = self.agent.neighbors_exchange(basis_dict)
        constraints = []
        for neigh in data:
            constraints.append(self.dict_to_constr(data[neigh]))

        const_list = [item for sublist in constraints for item in sublist]
        for con in self.agent.problem.constraints:
            const_list.append(con)

        p_tmp = Problem(self.agent.problem.objective_function, const_list)

        self.x = p_tmp.solve()
        self.B = self.evaluate_basis(const_list)

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

    def __init__(self, agent: Agent, enable_log: bool = False,
                 stopping_criterion: bool = False, stopping_iteration: int = 0,
                 is_standardform: bool = True, big_M: float = 500.0):

        self.agent = agent
        self.enable_log = enable_log
        self.sequence = None
        self.stopping_criterion = stopping_criterion
        self.stopping_iteration = stopping_iteration

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
        self.b_init = b_tmp[:, None]
        self.c_init = c_tmp
        if self.is_standardform:
            shape = self.A_init.shape[0]
        else:
            shape = self.A_init.shape[1]
        if self.is_standardform:
            H_data = np.vstack((c_tmp, A_tmp))
            BigM_H = create_bigM(H_data.shape[0]-1, self.big_M,
                                 self.is_standardform)
            H_data = np.hstack((H_data, BigM_H))
            b = self.b_init
        else:
            H_data = tostandard(A_tmp, b_tmp)
            BigM_H = create_bigM(H_data.shape[0]-1, self.big_M,
                                 True)
            H_data = np.hstack((H_data, BigM_H))
            b = self.c_init.transpose()
        init_lfbasis = gen_lexfeasible_basis(b, self.big_M)
        init_data = init_lexsimplex(H_data, init_lfbasis, shape)
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
            H_data = np.ndarray((self.B.shape[0], 0))
        else:
            H_data = np.ndarray((0, self.B.shape[1]))
        for neigh in data:
            self.x_neigh[neigh] = data[neigh]
            if self.is_standardform:
                H_data = np.hstack((H_data, self.x_neigh[neigh]))
            else:
                H_data = np.vstack((H_data, self.x_neigh[neigh]))
        if self.is_standardform:
            H_init = np.vstack((self.c_init, self.A_init))
            BigM_H = create_bigM(np.max(self.shape), self.big_M,
                                 self.is_standardform)
            H_data = np.hstack((H_data, H_init))
            H_data = np.hstack((H_data, BigM_H))
            BB = self.B
            b = self.b_init
        else:
            H_init = np.hstack((self.A_init, self.b_init))
            BigM_H = create_bigM(np.max(self.shape), self.big_M,
                                 self.is_standardform)
            H_data = np.vstack((H_data, H_init))
            H_data = tostandard(H_data[:, 0:-1], H_data[:, -1])
            BigM_H = create_bigM(H_data.shape[0]-1, self.big_M,
                                 True)
            H_data = np.hstack((H_data, BigM_H))
            BB = tostandard(self.B[:, 0:-1], self.B[:, -1])
            b = self.c_init.transpose()

        init_data = init_lexsimplex(H_data, BB, self.shape)

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


def create_bigM(nvar: int, bigM: float, is_standardform: bool):
    """Construct a Big-M matrix

    Args:
        nvar (int): LP size
        bigM (float): Big-M value
        is_standardform (bool): True if LP is in standard form

    Returns:
        numpy.ndarray: Big-M matrix
    """
    A_big = np.eye(nvar)

    if is_standardform:
        c_big = bigM*np.ones((1, nvar))
        BigM_H = np.vstack((c_big, A_big))
    else:
        A_big = np.vstack((A_big, -np.eye(nvar)))
        b_l = bigM*np.ones((nvar, 1))
        b_big = np.vstack((b_l, -bigM*np.ones((nvar, 1))))
        BigM_H = np.hstack((A_big, b_big))

    return BigM_H


def lexnegative(vector: np.ndarray, tol: float):
    """tests if a vector is lex-negative, i.e. , if its first
        non-zero component is negatve

    Args:
        vector (np.ndarray): input vector
        tol (float): numerical tolerance

    Returns:
        bool: True if vector is lexnegative
    """

    v = vector.flatten()
    s = vector.size

    zero_el = np.count_nonzero(v)
    # test if vector = 0
    if zero_el == 0:
        return False

    # Fix small digits
    for i in range(s):
        if (abs(v[i]) < tol):
            v[i] = 0

    for elem in v:
        # skip null components
        if elem == 0:
            continue

        # if the component is positive, return false
        if elem > 0:
            return False

        # else return true
        break

    return True


def lexnonpositive(vector: np.ndarray, tol: float):
    """tests if a vector is lex-negative, i.e. , if its first
        non-zero component is non-positive

    Args:
        vector (np.ndarray): input vector
        tol (float): numerical tolerance

    Returns:
        bool: True if vector is non-positive
    """
    s = vector.size

    zero_el = np.count_nonzero(vector)
    # test if vector = 0
    if zero_el == 0:
        return True

    for elem in vector:
        if abs(elem) < tol:  # Treat it as a zero
            continue
        # if the component is positive return false
        if elem > 0:
            return False

        # else return true
        break

    return True


def enteringcolumn(tableau, base_var, tol):
    """Simplex routine that selects an entering column

    Args:
        tableau (np.ndarray): standard form tableau
        base_var (np.ndarray): index of basic columns
        tol (float): numerical tolerance

    Returns:
        int: entering column
    """

    base_var_sort = np.sort(base_var.flatten())

    s = tableau.shape
    k = np.eye(s[1], s[1])
    I = k
    A = tableau[1:, :]
    B = A[:, base_var_sort]
    k[:, base_var_sort.tolist()] = I[:, base_var_sort.tolist()] - \
        np.linalg.solve(B, A).transpose()
    cost = tableau[0, :].transpose()
    reduced_cost = k @ cost

    kc_k_matrix = np.column_stack((reduced_cost, k))  # Algorithm 2 in Colin

    for ii in range(s[1]):
        # skip base variables
        count = 0
        for jj in base_var:
            if jj == ii:
                count += 1

        if count != 0:
            continue

        test_vector = kc_k_matrix[ii, :]

        if lexnonpositive(test_vector, tol):
            index = ii
            e = tableau[:, ii]
            return index

    return -1


def leavingcolumn(index_e, b, tableau, tol, base_var):
    """Simplex routine that selects a leaving column

    Args:
        index_e (int): current entering column
        b (np.ndarray): LP rhs
        tableau (np.ndarray): standard form tableau
        tol (float): numerical tolerance
        base_var (np.ndarray): index of basic columns

    Returns:
        int: leaving column
    """

    # Evaluate basis from tableau
    base = tableau[:, base_var]
    Ab = base[1:, :]
    # Inverse of basis
    Abinv = np.linalg.inv(Ab)
    ab_tmp = np.linalg.solve(Ab, b)
    M = np.column_stack((ab_tmp, Abinv))

    A = tableau[1:, :]
    v = np.linalg.solve(Ab, A)
    v = v[:, index_e]

    rows_m = []
    indices = []

    for jj in range(len(b)):
        if v[jj] > tol:
            rapporto = M[jj, :]/v[jj]
            rows_m.append(rapporto)
            indices.append(jj)

    rows_m = np.asarray(rows_m)
    indices = np.asarray(indices)
    if indices.size == 0:
        return -1

    index_sort = np.argsort(rows_m[:, 0])
    sorted_rows = rows_m[index_sort, :]

    tolerance_loc = tol

    for ijk in range(index_sort.size):
        l = indices[index_sort[ijk]]
        base_var_new = base_var
        base_var_new[l] = index_e
        base_var_new = np.sort(base_var_new.flatten())
        base_new = tableau[1:, base_var_new]
        beta_mat = np.linalg.inv(base_new)
        ab_tmp = np.linalg.solve(base_new, b)
        check_feas_mat = np.column_stack((ab_tmp, beta_mat))
        n_rows = check_feas_mat.shape[0]
        test_vector = np.zeros((n_rows, 1))
        for i_rows in range(n_rows):
            v_test = check_feas_mat[i_rows, 0]
            test_vector[i_rows] = lexnegative(v_test, tolerance_loc)
        if np.count_nonzero(test_vector) == 0:
            break

    return l


def simplex(tableau, b, base_var, nmax, tol_ent, tol_leav):

    flag_null = 0
    for nn in range(nmax):

        index_EC = enteringcolumn(tableau, base_var, tol_ent)
        if index_EC == -1:
            base_var = np.sort(base_var)
            return flag_null, base_var, nn

        base_var = np.sort(base_var.flatten())
        l = leavingcolumn(index_EC, b, tableau, tol_leav, base_var)

        if l == -1:
            flag_null = 1
            base_var = np.zeros((1, Ncon))
            return flag_null, base_var, nn

        base_var[l] = index_EC

    return flag_null, base_var, nn


def uniquebasis(H, B, unique_tol):
    H_urow = np.vstack(list({tuple(row) for row in H.transpose()}))
    H_ucol = H_urow.transpose()
    sizeH = H_ucol.shape

    H_uniq = np.ndarray((sizeH[0], 0))
    for ii in np.arange(sizeH[1]):
        # select cols of H
        i_colH = H_ucol[:, ii]
        # if one element of this is 1 then one constr. is already in Basis
        check_equality = np.zeros((sizeH[0]-1, 1))
        # loop on Basis cols
        for jj in np.arange(sizeH[0]-1):
            j_colB = B[:, jj]
            # if any element of this is 1 there is a difference btw the constr.
            check_equality_col = np.zeros((sizeH[0], 1))
            # loop on elements
            for kk in np.arange(len(i_colH)):
                if abs(i_colH[kk]-j_colB[kk]) > unique_tol:
                    check_equality_col[kk] = 1
            zero_elc = np.count_nonzero(check_equality_col)
            if zero_elc == 0:
                check_equality[jj] = 1
        zero_elb = np.count_nonzero(check_equality)
        if zero_elb == 0:
            H_uniq = np.hstack((H_uniq, i_colH[:, None]))
    H_uniq = np.hstack((B, H_uniq))
    return H_uniq


def init_lexsimplex(H_data, B, shape):
    H = uniquebasis(H_data, B, 1e-6)
    bas = np.arange(np.max(shape))

    H_t = H.transpose()
    ind = np.lexsort(np.flip(H_t.transpose(), 0))
    H_tsort = H_t[ind]

    H_sort = H_tsort.transpose()
    bas_sort = np.arange(np.max(shape))

    for i in np.arange(np.max(shape)):
        for j in np.arange(len(ind)):
            if bas[i] == ind[j]:
                bas_sort[i] = j
    bas_sort = np.sort(bas_sort)
    return H_sort, bas_sort


def tostandard(A, b):
    A_tmp = -A
    H = np.vstack((b.flatten(), A_tmp.transpose()))
    return H


def fromstandard(H, shape):
    A_B = H[1:, :]
    c_B = H[0, :]

    A = -A_B.transpose()
    b = np.reshape(c_B, shape)
    return A, b


def retrievelexsol(basic_tableau, b, is_standardform, shape):
    if is_standardform:
        B = basic_tableau
        A_B = basic_tableau[1:, :]
        x = np.reshape(np.linalg.solve(A_B, b), shape)
    else:
        b_data = fromstandard(basic_tableau, shape)
        B = np.hstack((b_data[0], b_data[1][:, None]))
        x = np.reshape(np.linalg.solve(b_data[0], b_data[1]), shape)
    return B, x


def gen_lexfeasible_basis(b, bigM):
    bflat = b.flatten()
    dim = np.max(b.shape)

    lex_feas_bas = np.ndarray((dim+1, 0))
    id_m = np.eye(dim)
    c_row = bigM*np.ones((1, 1))

    for i in np.arange(dim):
        if bflat[i] > 0:
            col_i = id_m[i, :][:, None]
        else:
            col_i = -id_m[i, :][:, None]
        col_i = np.vstack((c_row, col_i))
        lex_feas_bas = np.hstack((lex_feas_bas, col_i))
    return lex_feas_bas
