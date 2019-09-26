Special Functions
==============================

Stochastic Function
-----------------------------------------

.. autoclass:: disropt.functions.stochastic_function.StochasticFunction
    :members:
    :exclude-members: eval, subgradient, jacobian
    :undoc-members:
    :show-inheritance:

    .. automethod:: eval(self, w)
    .. automethod:: jacobian(self, x)
    .. automethod:: subgradient(self, x)


Function With Extended Variable
-----------------------------------------

.. autoclass:: disropt.functions.extended_function.ExtendedFunction
    :members:
    :exclude-members: eval, subgradient, jacobian
    :undoc-members:
    :show-inheritance:
