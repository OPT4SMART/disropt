.. _advanced_problems:

Optimization Problems
========================

The :class:`Problem` class allows one to define optimization problems of various types. Consider the following problem:

.. math::

    \text{minimize } & \| A^\top x - b \|

    \text{subject to } & x \geq 0

with :math:`x\in\mathbb{R}^4`. We can define it as::

    import numpy as np
    from disropt.problems import Problem
    from disropt.functions import Variable, Norm

    x = Variable(4)
    A = np.random.randn(n, n)
    b = np.random.randn(n, 1)

    obj = Norm(A @ x - b)
    constr = x >= 0

    pb = Problem(objective_function = obj, constraints = constr)

.. In the distributed framework of **disropt**, the :class:`Problem` class is mainly meant to define portions of a bigger optimization problem that are locally known to local computing units (see the following tutorial).

.. However, since in many distributed optimization algorithms can be requested to solve local optimization problems, we implemented some problem solvers.

If the problem is convex, it can be solved as::

    solution = pb.solve()

Generic (convex) nonlinear problems of the form

.. math::

    \text{minimize } & f(x)

    \text{subject to } & g(x) \leq 0

                        & h(x) = 0


are solved through the cvxpy_ solver (when possible), or with the cvxopt_ solver, while more structured problems (LPs and QPs) can be solved through other solvers (osqp_ and glpk_). The integration with other solvers will be provided in future releases.
LPs and QPs can be directly defined through specific classes (:class:`LinearProblem` and :class:`QuadraticProblem`). However, the :class:`Problem` class is capable to recognize LPs and QPs, which are automatically converted into the appropriate format.

Projection onto the constraints set
-------------------------------------------

Projecting a point onto the constraints set of a problem is often required in distributed optimization algorithms. The method :class:`project_on_constraint_set` is available to do this::

    projected_point = pb.project_on_constraint_set(pt)


.. _cvxpy: http://cvxpy.org
.. _cvxopt: http://cvxopt.org
.. _osqp: https://osqp.org
.. _glpk: https://www.gnu.org/software/glpk/