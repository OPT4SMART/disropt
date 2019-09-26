.. _tutorial_functions_constraints:

====================================
Objective functions and constraints
====================================

Functions
--------------------------------
**disropt** comes with many already implemented mathematical functions.
Functions are defined in terms of optimization variables (:class:`Variable`) or other functions.
Let us start by defining a :class:`Variable` object as::

    from disropt.functions import Variable
    n = 2 # dimension of the variable
    x = Variable(n)
    print(x.input_shape) # -> (2, 1)
    print(x.output_shape) # -> (2, 1)

Now, suppose you want to define an affine function :math:`f(x)=A^\top x - b` with :math:`A\in\mathbb{R}^{2\times 2}` and :math:`b\in\mathbb{R}^2`::

    import numpy as np
    a = 1
    A = np.array([[1,2], [2,4]])
    b = np.array([[1], [1]])
    f = A @ x - b

    # or, alternatively
    from disropt.functions import AffineForm
    f = AffineForm(x, A, b)

The composition of functions is fully supported. Suppose you want to define a function :math:`g(x)=f(x)^\top Q f(x)`, then::

    from disropt.functions import QuadraticForm
    Q = np.random.rand(2,2)
    g = QuadraticForm(f, Q) # or: g = f @ (Q.tranpose() @ f)
    print(g.input_shape) # -> (2, 1)
    print(g.output_shape) # -> (1, 1)

Currently supported operations with functions are sum (`+`), difference (`-`), product(`*`) and matrix product (`@`). Combination with numpy arrays is supported as well.


Function properties and methods
+++++++++++++++++++++++++++++++++

Each function has three properties that can be checked: differentiablity, being affine and quadratic::

    g.is_differentiable # -> True
    g.is_affine # -> False
    g.is_quadratic # -> True
    f.is_affine # -> True

and their input and output shapes can be obtained as ::

    g.output_shape # -> (1,1)
    g.input_shape # -> (2,1)

Moreover, it is possible to evaluate functions at desired points and to obtain the corresponding (sub)gradient/jacobian/hessian as::

    pt = np.random.rand(2,1)
    # the value of g computed at pt is obtained as
    g.eval(pt) 
    # the value of the jacobian of g computed at pt is
    g.jacobian(pt) 
    # the value of a (sub)gradient of g is available only if the output shape of g is (1,1)
    g.subgradient(pt) 
    # otherwise it will result in an error
    f.subgradient(pt) # -> Error
    # the value of the hessian of g computed at pt is
    g.hessian(pt) 

For affine and quadratic functions, a method called :class:`get_parameters` is implemented, which returns the matrices and vectors that define those functions.
The generic form for an affine function is :math:`A^\top x + b` while the one for a quadratic form is :math:`x^\top P x + q^\top x + r`::

    f = A @ x + b
    f.get_parameters() # -> A, b

Defining constraints from functions
-----------------------------------

Constraints are represented in the canonical forms :math:`f(x)=0` and :math:`f(x)\leq 0`.

They are directly obtained from functions::

    constraint = g == 0 # g(x) = 0
    constraint = g >= 0 # g(x) >= 0
    constraint = g <= 0 # g(x) <= 0

On the right side of (in)equalities, numpy arrays and functions (with appropriate shapes) are also allowed::

    c = np.random.rand(2,1)
    constr = f <= c

which is automatically translated in the corresponding canonical form.

Constraints can be evaluated at any point by using the :class:`eval` method which returns a boolean value if the constraint
is satisfied. Moreover, the function defining a constraints can be retrieved with the :class:`function` method::

    pt = np.random.rand(2,1)
    constr.eval(pt) # -> True if f(pt) <= c
    constr.function.eval(pt) # -> value of f - c at pt

Affine and quadratic constraints
+++++++++++++++++++++++++++++++++

Parameters defining affine and quadratic constraints can be easily obtained. They can be accessed by calling the :class:`get_parameters` method::

    f = A @ x + b
    constraint = f == 0 # affine equality constraint
    # f has the form A^T x + b
    constraint.get_parameters() # returns A and b

    g = f @ f
    constraint = g == 0 # quadratic equality constraint
    # g has the form x^T P x + q^T x + r
    constraint.get_parameters() # returns P, q and r

Projection onto a constraint set
+++++++++++++++++++++++++++++++++
The projection of a point onto the set defined by a constraint can be computed via the :class:`projection` method::

    projected_point = f.projection(pt) 

Constraint sets
+++++++++++++++++++++++++++++++++

Some particular constraint sets (for which projection of points is easy to compute)
are also available through specialized classes, which are extensions of the class :class:`Constraint`.
For instance, suppose you want all the components of :math:`x` to be in :math:`[-1,1]`.
Then you can define a :class:`Box` constraint as::

    from disropt.constraints import Box
    bound = np.ones((2,1))
    constr = Box(-bound, bound)

Two methods are available: :class:`projection` and :class:`intersection`.
The first one returns the projection of a given point on the set,
while the second one intersects the set with another one.
This feature is particularly useful in set-membership estimation algorithms.

Constraint sets can be converted into a list of constraints through the method :code:`to_constraints`.
