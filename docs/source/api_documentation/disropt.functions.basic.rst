Basic Functions
==============================

Here is the list of the implemented basic mathematical functions.


Variable
-----------------------------------------

.. autoclass:: disropt.functions.variable.Variable
    :members:
    :exclude-members: eval, subgradient, jacobian
    :undoc-members:
    :show-inheritance:

    .. automethod:: eval(self, w)


AffineForm
-----------------------------------------

.. autoclass:: disropt.functions.affine_form.AffineForm
    :members:
    :exclude-members: eval
    :undoc-members:
    :show-inheritance:

    .. automethod:: eval(self, w)


QuadraticForm
-----------------------------------------

.. autoclass:: disropt.functions.quadratic_form.QuadraticForm
    :members:
    :exclude-members: eval, subgradient, jacobian
    :undoc-members:
    :show-inheritance:

    .. automethod:: eval(self, w)


Abs
-----------------------------------------

.. autoclass:: disropt.functions.abs.Abs
    :members:
    :exclude-members: eval, subgradient, jacobian
    :undoc-members:
    :show-inheritance:

    .. automethod:: eval(self, w)


Norm
-----------------------------------------

.. autoclass:: disropt.functions.norm.Norm
    :members:
    :exclude-members: eval, subgradient, jacobian
    :undoc-members:
    :show-inheritance:

    .. automethod:: eval(self, w)


SquaredNorm
-----------------------------------------

.. autoclass:: disropt.functions.squared_norm.SquaredNorm
    :members:
    :exclude-members: eval, subgradient, jacobian
    :undoc-members:
    :show-inheritance:

    .. automethod:: eval(self, w)

Log
-----------------------------------------

.. autoclass:: disropt.functions.log.Log
    :members:
    :exclude-members: eval, subgradient, jacobian
    :undoc-members:
    :show-inheritance:

    .. automethod:: eval(self, w)

Exp
-----------------------------------------

.. autoclass:: disropt.functions.exp.Exp
    :members:
    :exclude-members: eval, subgradient, jacobian
    :undoc-members:
    :show-inheritance:

    .. automethod:: eval(self, w)

Logistic
-----------------------------------------

.. autoclass:: disropt.functions.logistic.Logistic
    :members:
    :exclude-members: eval, subgradient, jacobian
    :undoc-members:
    :show-inheritance:

    .. automethod:: eval(self, w)

Min
-----------------------------------------

.. autoclass:: disropt.functions.min.Min
    :members:
    :exclude-members: eval, subgradient, jacobian
    :undoc-members:
    :show-inheritance:

    .. automethod:: eval(self, w)


Max
-----------------------------------------

.. autoclass:: disropt.functions.max.Max
    :members:
    :exclude-members: eval, subgradient, jacobian
    :undoc-members:
    :show-inheritance:

    .. automethod:: eval(self, w)

Square
-----------------------------------------

.. autoclass:: disropt.functions.square.Square
    :members:
    :exclude-members: eval, subgradient, jacobian
    :undoc-members:
    :show-inheritance:

    .. automethod:: eval(self, w)
