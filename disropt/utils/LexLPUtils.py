import numpy as np


def create_bigM(nvar, bigM, is_standardform):

    A_big = np.eye(nvar)

    if is_standardform:
        c_big = bigM*np.ones((1, nvar))
        BigM_H = np.vstack((c_big, A_big))
    else:
        A_big = np.vstack((A_big, -eye(nvar)))
        b_l = bigM*np.ones((nvar, 1))
        b_big = np.vstack((b_l, -bigM*ones((nvar, 1))))
        BigM_H = np.hstack((A_big, b_big))

    return BigM_H


def lexnegative(vector, tol):

    # LEXNEGATIVE: tests if a vector is lex-negative, i.e. , if its first
    # non-zero component is negatve
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


def lexnonpositive(vector, tol):

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


def enteringcolumn(c, tableau, base_var, tol):
    # ENTERINGCOLUMN: selects a non-basic entering column which can enter the
    # basis.

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


def leavingcolumn(c, index_e, b, tableau, tolerance, base_var):

    # % LEAVINGCOLUMN: selects the basic column that has to be exchanged with
    # % a non-basic enetering column in the Pivot operation

    # % Requires:
    # % c: dimension of the basis
    # % index: index of entering column
    # % base: current basis
    # % b: rhs of Ax=b, i.e., vector of ones representing the equality
    # % tableau: (c+1) x k matrix with costs (first rows) and constraints
    # % constraints

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

    for jj in range(c):
        if v[jj] > tolerance:
            rapporto = M[jj, :]/v[jj]
            rows_m.append(rapporto)
            indices.append(jj)

    rows_m = np.asarray(rows_m)
    indices = np.asarray(indices)

    if indices.size == 0:
        return -1

    index_sort = np.argsort(rows_m[:, 0])
    sorted_rows = rows_m[index_sort, :]

    tolerance_loc = tolerance

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
    Ncon = len(b)
    flag_null = 0
    for nn in range(nmax):

        index_EC = enteringcolumn(Ncon, tableau, base_var, tol_ent)
        if index_EC == -1:
            base_var = np.sort(base_var)
            return flag_null, base_var, nn

        base_var = np.sort(base_var.flatten())
        l = leavingcolumn(Ncon, index_EC, b, tableau, tol_leav, base_var)

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
    H = np.vstack(b.flatten(), A_tmp.transpose())
    return H


def fromstandard(H, shape):
    A_B = H[1:, :]
    c_B = H[0, :]

    A = -A_B.transpose()
    b = np.reshape(c_b, shape)
    return A, b


def retrievelexsol(basic_tableau, b, is_standardform, shape):
    if is_standardform:
        B = basic_tableau
        A_B = basic_tableau[1:, :]
        x = np.reshape(np.linalg.solve(A_B, b), shape)
    else:
        b_data = fromstandard(basic_tableau, shape)
        B = hstack(b_data[0], b_data[1])
        x = np.reshape(np.linalg.solve(b_data[0], b_data[1]), shape)
    return B, x
