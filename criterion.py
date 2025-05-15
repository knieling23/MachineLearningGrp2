import unittest
import numpy as np

"""
Dieses Modul beinhaltet Klassen die verschiedene Loss-Functions implementieren.
Jede Methode einer Klasse gibt das Ergebnis direkt zurück, als einzige Klassenvariable
ist der Regularisationsfaktor gespeichert.
Methoden beginnent mit einem Unterstrich, sind als private Methoden zu verstehen und
sollten nur innerhalb der Klasse verwendet werden.
"""
class mse_loss():
    """
    Diese Klasse implementiert die Mean-Squared-Error Loss-Function.
    """
    def __init__(self, reg):
        self.reg = reg

    def _mean_squared_error(self, prediction, correct_output):
        return np.mean((prediction - correct_output) ** 2)

    def _calc_data_loss(self, prediction, correct_output):
        return np.mean((prediction - correct_output) ** 2)

    def _calc_reg_loss(self, weights):
        return int(np.sum(weights ** 2) / 4)

    def calc_loss(self, prediction, correct_output, weights):
        data_loss = self._calc_data_loss(prediction, correct_output)
        reg_loss = self._calc_reg_loss(weights)
        return data_loss + 2 * self.reg * reg_loss

    def gradient_data_loss(self, prediction, correct_output):
        return 2 * (prediction - correct_output) / prediction.shape[0]

    def gradient_reg(self, weights):
        return self.reg * weights

class softmax_cross_entropy_loss():
    """
    Diese Klasse implementiert die Softmax Cross-Entropy Loss-Function.
    """
    def __init__(self, reg):
        self.reg = reg

    def _softmax(self, prediction):
        exp_scores = np.exp(prediction - np.max(prediction, axis=1, keepdims=True))
        return exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
    
    def _cross_entropy(self, prediction, correct_output):
        probs = self._softmax(prediction)
        log_likelihood = np.log(probs[np.arange(len(correct_output)), correct_output])
        return np.mean(log_likelihood)

    def _calc_data_loss(self, prediction, correct_output):
        probs = self._softmax(prediction)
        log_likelihood = -np.log(probs[range(len(correct_output)), correct_output])
        return np.mean(log_likelihood)

    def _calc_reg_loss(self, weights):
        return int(np.sum(weights ** 2) / 4)

    def calc_loss(self, prediction, correct_output, weights):
        data_loss = self._calc_data_loss(prediction, correct_output)
        reg_loss = self._calc_reg_loss(weights)
        return data_loss + 2 * self.reg * reg_loss

    def gradient_data_loss(self, prediction, correct_output):
        num_samples = prediction.shape[0]
        probs = self._softmax(prediction)
        probs[range(num_samples), correct_output] -= 1
        return probs / num_samples

    def gradient_reg(self, weights):
        return self.reg * weights
class TestClass_mse_loss(unittest.TestCase):
    def setUp(self):
        self.c = mse_loss(0.5)

    def test__mean_squared_error(self):
        i = np.array([[1,2,3], [4,5,6]])
        c = np.array([[2, 0, 3], [2, 0, 3]])
        o = 7.166666666666667
        self.assertEqual(self.c._mean_squared_error(i,c), o)

    def test__calc_data_loss(self):
        i = np.array([[2,2,8], [4,4,16]])
        c = np.array([[6, 4, 4], [2, 1, 1]])
        o = 45.666666666666664
        self.assertEqual(self.c._calc_data_loss(i,c), o)

    def test__calc_reg_loss(self):
        i = np.array([[2,2,8], [4,4,16]])
        o = 90
        self.assertEqual(self.c._calc_reg_loss(i), o)

    def test_calc_loss(self):
        i = np.array([[2,6,8], [4,5,16]])
        w = np.array([[2,2,8], [4,4,16]])
        c = np.array([[6, 4, 4], [2, 2, 2]])
        o = 130.83333333333334
        self.assertEqual(self.c.calc_loss(i,c,w), o)

    def test_gradient_data_loss(self):
        i = np.array([[2,2,8], [4,4,16]]).astype(np.float64)
        c = np.array([[6, 4, 4], [2, 1, 1]]).astype(np.float64)
        o = np.array([[-4.0, -2.0, 4.0], [2.0, 3.0, 15.0]]).astype(np.float64)
        self.assertEqual(self.c.gradient_data_loss(i,c).tolist(), o.tolist())

    def test_gradient_reg(self):
        i = np.array([[2,2,8], [4,4,16]])
        o = [[1.0, 1.0, 4.0], [2.0, 2.0, 8.0]]
        self.assertEqual(self.c.gradient_reg(i).tolist(), o)

class TestClass_softmax_cross_entropy_loss(unittest.TestCase):
    def setUp(self):
        self.c = softmax_cross_entropy_loss(0.5)

    def test__softmax(self):
        num_samples = 5
        self.assertEqual(np.sum(self.c._softmax(np.random.rand(num_samples,2))), num_samples)

    def test__cross_entropy(self):
        i = np.array([[1,2,3], [4,5,6]])
        c = np.array([2, 0])
        o = -1.2424533248940002
        self.assertEqual(self.c._cross_entropy(i,c), o)

    def test__calc_data_loss(self):
        i = np.array([[1,2,3], [4,5,6]])
        c = np.array([2, 0])
        o = 1.4076059644443804
        self.assertEqual(self.c._calc_data_loss(i,c), o)

    def test__calc_reg_loss(self):
        i = np.array([[2,2,8], [4,4,16]])
        o = 90
        self.assertEqual(self.c._calc_reg_loss(i), o)

    def test_calc_loss(self):
        i = np.array([[2,6,8], [4,5,16]])
        w = np.array([[2,2,8], [4,4,16]])
        c = np.array([0, 1])
        o = 98.56456587724102
        self.assertEqual(self.c.calc_loss(i,c,w), o)

    def test_gradient_data_loss(self):
        i = np.array([[2,2,8], [4,4,16]]).astype(np.float64)
        c = np.array([1, 1])
        o = np.array([[0.0012332621856788754, -0.4987667378143211, 0.49753347562864225],
                      [3.072068425782561e-06, -0.49999692793157424, 0.4999938558631484]]).astype(np.float64)
        self.assertEqual(self.c.gradient_data_loss(i,c).tolist(), o.tolist())

if __name__ == '__main__':
    unittest.main()
