import numpy as np
from autograd import grad as ag_grad
from autograd import jacobian as ag_jacobian
from autograd import hessian as ag_hessian
import warnings
from typing import List
from .abstract_function import AbstractFunction
from .utilities import check_input


class StochasticFunction(AbstractFunction):
    """Stochastic function

    .. math::

        f(x)=\\mathbb{E}[h(x)]

    with :math:`x: \\mathbb{R}^{n}`.

    The :class:`random_batch` method extract a batch from the function and :class:`batch_subgradient`, :class:`batch_jacobian` and :class:`batch_hessian` methods return a subgradient, jacobian and hessian computed at on the last batch.

    Args:
        fn_list (list): list of AbstractFunction objects
        probabilities (list, optional): list with the probabilities of drawing each function. Default is None which leads to uniform probabilities

    Raises:
        TypeError: fn_list input must be a list of functions
        ValueError: All functions must have the same input/output shape
        TypeError: probabilities argument must be a list of floats
        ValueError: inputs must have the same lenght
        ValueError: provided probabilities must sum to 1
        NotImplementedError: only 1, 2 and inf norms are currently supported
    """

    def __init__(self, fn_list: List[AbstractFunction], probabilities: List[float]):
        if not isinstance(fn_list, list):
            raise TypeError("fn_list input must be a list of functions")
        for f in fn_list:
            if not isinstance(fn, AbstractFunction):
                raise TypeError("fn_list input must be a list of functions")

        input_shape = fn_list[0].input_shape
        output_shape = fn_list[0].output_shape
        self.differentiable = True
        self.affine = True
        self.quadratic = True
        for item in fn_list:
            if item.input_shape != input_shape or item.output_shape != output_shape:
                raise ValueError("All functions must have the same input/output shape")
            if not item.is_differentiable:
                self.differentiable = False
            if not item.is_affine:
                self.affine = False
            if not item.is_quadratic:
                self.quadratic = False

        self.fn_list = fn_list
        self.items = len(fn_list)
        self.input_shape = input_shape
        self.output_shape = output_shape

        if probabilities is not None:
            if not isinstance(probabilities, list):
                raise TypeError("probabilities argument must be a list")
            for p in probabilities:
                if not isinstance(p, float):
                    raise TypeError("probabilities argument must be a list")
            if len(probabilities) != len(fn_list):
                raise ValueError("inputs must have the same lenght")
            if sum(probabilities) != 1.0:
                raise ValueError("provided probabilities must sum to 1")
        else:
            probabilities = (np.ones(self.items)/self.items).tolist()
        self.probabilities = probabilities

        super().__init__()

    @check_input
    def eval(self, x: np.ndarray) -> np.ndarray:
        value = 0
        for item in range(self.items):
            value += self.probabilities[item] * self.fn_list[item].eval(x)
        return value.reshape(self.output_shape)

    def random_batch(self, batch_size: int = 1):
        """generate a random batch from the function.
        
        Args:
            batch_size (int, optional): batch size. Defaults to 1.
        """
        self.selected = np.random.choice(self.items,
                                         size=batch_size,
                                         p=self.probabilities)
        self._setbatchdiff()

    def _setbatchdiff(self):
        self._batch_subgradient = ag_grad(self._batch_eval)
        self._batch_jacobian = ag_jacobian(self._batch_eval)
        self._batch_hessian = ag_hessian(self._batch_eval)
    
    def _batch_eval(self, x: np.ndarray) -> np.ndarray:
        value = 0
        for sample in self.selected:
            value += self.fn_list[sample].eval(x)
        return value.reshape(self.output_shape)

    @check_input
    def batch_jacobian(self, x: np.ndarray):
        """evaluate the jacobian on the current batch
        
        Args:
            x (np.ndarray): point
        Returns:
            numpy.ndarray: jacobian
        """
        return np.squeeze(self._batch_jacobian(x))
    
    @check_input
    def batch_hessian(self, x: np.ndarray):
        """evaluate the hessian on the current batch
        
        Args:
            x (np.ndarray): point
        
        Returns:
            numpy.ndarray: hessian
        """
        return np.squeeze(self._batch_hessian(x))

    @check_input
    def batch_subgradient(self, x: np.ndarray):
        """evaluate the subgradient on the current batch
        
        Args:
            x (np.ndarray): point
        
        Raises:
            ValueError: Only functions with scalar output have a subgradient
        
        Returns:
            numpy.ndarray: subgradient
        """
        if not self.output_shape == (1, 1):
            raise ValueError("Undefined subgradient for output_shape={}".format(output_shape))
        else:
            return self._batch_subgradient(x).reshape(self.input_shape)


    # @check_input
    # def jacobian(self, x: np.ndarray, batch_size: int = 1, **kwargs) -> np.ndarray:
    #     selected = np.random.choice(self.items,
    #                                 size=batch_size,
    #                                 p=self.probabilities)

    #     jac = self.fn_list[selected[0]].jacobian(x)
    #     if selected.size > 1:
    #         for sel in selected[1:]:
    #             jac += self.fn_list[sel].jacobian(x)

    #     return jac
