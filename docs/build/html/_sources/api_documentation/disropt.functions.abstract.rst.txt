Abstract Function
==============================

AbstractFunction
-----------------------------------------

.. autoclass:: disropt.functions.abstract_function.AbstractFunction
    :members:
    :exclude-members: eval, subgradient, jacobian
    :undoc-members:
    :show-inheritance:

    .. automethod:: eval(self, w)
    .. automethod:: jacobian(self, x)
    .. automethod:: subgradient(self, x)