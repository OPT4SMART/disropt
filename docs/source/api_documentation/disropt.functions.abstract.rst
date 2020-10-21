Abstract Function
==============================

AbstractFunction
-----------------------------------------

.. autoclass:: disropt.functions.abstract_function.AbstractFunction
    :members:
    :exclude-members: eval, subgradient, jacobian, input_shape, output_shape, differentiable, affine, quadratic
    :undoc-members:
    :show-inheritance:

    .. automethod:: eval(self, w)
    .. automethod:: jacobian(self, x)
    .. automethod:: subgradient(self, x)