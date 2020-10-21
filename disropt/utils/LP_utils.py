import numpy as np
from ..problems.linear_problem import LinearProblem
from ..functions import Variable

def generate_LP(n_var: int, n_constr: int, radius: float, direction: str='min', constr_form: str='ineq'):
    """Generate a feasible and not unbounded Linear Program and return problem data (cost, constraints, solution)
    TODO add reference
    
    Args:
        n_var (int): number of optimization variables
        n_constr (int): number of constraints
        radius (float): size of feasible set (for constr_form = 'ineq'), size of dual feasible set (for constr_form = 'eq')
        direction (str, optional): optimization direction - either 'max' (for maximization) or 'min' (for minimization). Defaults to 'min'
        constr_form (str, optional): form of constraints - either 'ineq' (for inequality: :math:`Ax \le b`) or 'eq' (for standard form: :math:`Ax = b, x \ge 0`). Defaults to 'ineq'

    Returns:
        tuple:
            - **c** (*numpy.ndarray*): cost vector
            - **A** (*numpy.ndarray*): constraint matrix
            - **b** (*numpy.ndarray*): constraint vector (right-hand side)
            - **solution** (*numpy.ndarray*): optimal solution of problem
    """

    size_1 = n_constr
    size_2 = n_var
    x = Variable(n_var)

    # swap dimensions if in dual form (equality constraints)
    if constr_form == 'eq':
        size_1, size_2 = size_2, size_1

    count = 0

    while True:
        count += 1

        # generate a problem in the form min_y c'y s.t. Ay <= b
        A_gen = np.random.randn(size_1, size_2)
        b_gen = np.random.randn(size_1, 1)
        c_gen = A_gen.transpose() @ np.random.randn(size_1, 1)

        # prepare problem data
        if constr_form == 'ineq':
            # primal problem
            A = A_gen
            b = b_gen
            c = c_gen
            constr = A_gen.transpose() @ x <= b_gen
        else:
            # dual problem: min_x b'x s.t. A'x = -c, x >= 0
            A = A_gen.transpose()
            b = -c_gen
            c = b_gen
            constr = [A_gen @ x == -c_gen, x >= 0]

        # form optimization problem object
        if direction == 'min':
            obj = c @ x
        else:
            obj = -c @ x

        prob = LinearProblem(objective_function=obj, constraints=constr)

        # solve problem to check whether an optimal solution exists
        try:
            solution = prob.solve()
            break
        except:
            continue

    return c, A, b, solution
