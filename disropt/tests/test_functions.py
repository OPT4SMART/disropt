import unittest
import numpy as np
import warnings

from ..functions import *

class TestFunction(unittest.TestCase):
    """unit test for functions module"""

    def setUp(self):
        self.n = 3
        self.m = 2

        self.x = Variable(self.n)

        P = np.random.randn(self.n, self.n)
        self.P = P.transpose() @ P
        self.q = np.random.rand(self.n, 1)

        self.A = np.random.randn(self.n, self.m)
        self.b = np.random.rand(self.m, 1)
    
    def test_affine(self):
        warnings.simplefilter('ignore', category=UserWarning)
        affine_form = self.A @ self.x + self.b
        affine_form_implicit = AffineForm(self.x, self.A, self.b)

        self.assertEqual(affine_form, affine_form_implicit)
        pt = np.random.randn(*self.x.input_shape)
        np.testing.assert_almost_equal(affine_form.jacobian(pt), affine_form._alternative_jacobian(pt))
        self.assertIsInstance(affine_form @ affine_form, QuadraticForm)

if __name__ == '__main__':
    unittest.main()