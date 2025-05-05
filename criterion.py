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
        """Initialisierung.

        Parameter:
          reg - Der Regularisationsfaktor"""
        self.reg = reg

    def _mean_squared_error(self, prediction, correct_output):
        """
        Berechnet den Mean-Squared-Error.

        Parameter:
          prediction - Die berechnete Klassifizierung
          correct_output - Die korrekte Klassifizierung
        """
        pass
            
    def _calc_data_loss(self, prediction, correct_output):
        """
        Berechnet den Data-Loss einer Loss-Function.
        Für diese Klasse ist dies der MSE.

        Parameter:
          prediction - Die berechnete Klassifizierung
          correct_output - Die korrekte Klassifizierung
        """
        pass

    def _calc_reg_loss(self, weights):
        """
        Berechnet die L2 Regularization.
        https://towardsdatascience.com/l1-and-l2-regularization-methods-ce25e7fc831c
        
        Hinweis: Multiplizieren Sie einen Faktor von 0.5
                 hinzu. Dies wird bei der Ableitung helfen.

        Parameter:
          weights - Die Gewichte des Networks
        """
        pass

    def calc_loss(self, prediction, correct_output, weights):
        """
        Berechnet den Loss -> Summe aus Data-Loss und Regularization-Loss.

        Parameter:
          prediction - Die berechnete Klassifizierung
          correct_output - Die korrekte Klassifizierung
          weights - Die Gewichte des Networks
        """
        pass
    
    def gradient_data_loss(self, prediction, correct_output):
        """
        Berechnet die Ableitung des Data-Losses.

        Parameter:
          prediction - Die berechnete Klassifizierung
          correct_output - Die korrekte Klassifizierung
        """
        pass

    def gradient_reg(self, weights):
        """
        Berechnet die Ableitung des Regularization-Loss.
        Hinweis: Bedenken Sie den 0.5 Faktor in _calc_reg_loss
        
        Parameter:
          weights - Die Gewichte des Networks
        """
        pass


class softmax_cross_entropy_loss():
    """
    Diese Klasse implementiert die Softmax Cross-Entropy Loss-Function.
    """

    def __init__(self, reg):
        """Initialisierung.

        Parameter:
          reg - Der Regularisationsfaktor"""
        self.reg = reg

    def _softmax(self, prediction):
        """
        Berechnet den Softmax.
        https://towardsdatascience.com/softmax-activation-function-how-it-actually-works-d292d335bd78

        Parameter:
          prediction - Die berechnete Klassifizierung
        """
        pass

    def _cross_entropy(self, prediction, correct_output):
        """
        Berechnet die Average Cross-Entropy.
        https://towardsdatascience.com/cross-entropy-loss-function-f38c4ec8643e

        Parameter:
          prediction - Die berechnete Klassifizierung
          correct_output - Die korrekte Klassifizierung
        """
        pass

    def _calc_data_loss(self, prediction, correct_output):
        """
        Berechnet den Data-Loss einer Loss-Function.
        Für diese Klasse ist dies die Cross-Entropy des Softmaxes.

        Parameter:
          prediction - Die berechnete Klassifizierung
          correct_output - Die korrekte Klassifizierung
        """
        pass

    def _calc_reg_loss(self, weights):
        """
        Berechnet die L2 Regularization.
        https://towardsdatascience.com/l1-and-l2-regularization-methods-ce25e7fc831c
        
        Hinweis: Multiplizieren Sie einen Faktor von 0.5
                 hinzu. Dies wird bei der Ableitung helfen.

        Parameter:
          weights - Die Gewichte des Networks
        """
        pass
    
    def calc_loss(self, prediction, correct_output, weights):
        """
        Berechnet den Loss -> Summe aus Data-Loss und Regularization-Loss.

        Parameter:
          prediction - Die berechnete Klassifizierung
          correct_output - Die korrekte Klassifizierung
          weights - Die Gewichte des Networks
        """
        pass

    def gradient_data_loss(self, prediction, correct_output):
        """
        Berechnet die Ableitung des Data-Losses.
        https://towardsdatascience.com/derivative-of-the-softmax-function-and-the-categorical-cross-entropy-loss-ffceefc081d1
        Hinweis: Bedenken Sie, dass in _calc_data_loss der Average Cross Entropy berechnet wird.
                 Somit kommt noch ein Faktor zur Ableitung dazu.
        
        Parameter:
          prediction - Die berechnete Klassifizierung
          correct_output - Die korrekte Klassifizierung
        """
        pass

    def gradient_reg(self, weights):
        """
        Berechnet die Ableitung des Regularization-Loss.
        Hinweis: Bedenken Sie den 0.5 Faktor in _calc_reg_loss
        
        Parameter:
          weights - Die Gewichte des Networks
        """
        pass


"""
Ab hier nur Tests!
Für eigene Tests, erstellen Sie eine neue TestClass.
"""

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
        pass

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
