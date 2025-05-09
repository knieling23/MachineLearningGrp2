import numpy as np
import unittest
"""
Dieses Modul implementiert alle Optimizer.
"""


class bgd():
    """
    Diese Klasse implementiert den Batch Gradient Descent Optimizer.
    """
    
    def __init__(self, step_size):
        """
        Initialisierung.

        Parameter:
          step_size - Die Step-Size wie die Parameter verbessert werden
        """
        self.step_size = step_size

    def step(self, gradient, param):
        """
        Optimiert die übergebenen Parameter.

        Parameter:
          gradient - Der Berechnete Gradient
          param - Die Parameter die optimiert werden sollen

        Return:
          Die optimierten Parameter
        """
        return param - self.step_size * gradient


class TestCase_bgd(unittest.TestCase):

    def setUp(self):
        self.c = bgd(4)

    def test_step(self):
        g = np.array([1,2,3])
        p = np.array([4,5,7])
        o = [0, -3, -5]
        self.assertEqual(self.c.step(g,p).tolist(), o)

if __name__ == '__main__':
    unittest.main()
