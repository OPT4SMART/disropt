import numpy as np
import autograd
import autograd.numpy as anp
from functools import wraps  # AT: I added this line


# decorator for checking inputs to eval and subgradient


def check_input(checked_fn):
    @wraps(checked_fn)  # AT: I added this line
    def check(fn, x, **kwargs):
        if x is not None:
            if not isinstance(x, (np.ndarray, anp.numpy_boxes.ArrayBox)) or not (fn.input_shape == x.shape):
                raise ValueError(
                    "Input must be a numpy.ndarray with shape {}".format(
                        fn.input_shape))
        block = kwargs.get("block", None)
        if block is not None:
            if not isinstance(block, np.ndarray):
                raise TypeError("block argument must be a numpy.ndarray")
        return checked_fn(fn, x, **kwargs)
    return check
