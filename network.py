import unittest
import numpy as np
"""
Dieses Modul implementiert die Klasifizierer.
Jede Funktion liefert das Ergebnis direkt zurück.

*Ausschließlich der Gewichte und Biases, werden keine Werte in der Klasse gespeichert.*
"""


class linearClassifier():
    """
    Klasse die einen "simplen" Klassifizierer implementiert.
    """
    
    def __init__(self, dims, num_classes):
        """
        Initialisierung
        """
        self.weights = 0.01 * np.random.randn(dims,num_classes)
        self.bias = np.zeros((1,num_classes))

    def forward(self, data_input):
        """
        Berechnet die Forward-Step des Networks.

        Parameter:
          data_input - Input des Datasets
        """
        return np.dot(data_input, self.weights) + self.bias

    def backprop_layer_weights(self, data_input, score_gradients):
        """
        Backpropagates den Output durch den Layer bezüglich der Gewichte.

        Parameter:
          data_input - Input des Datasets
          score_gradients - Der Gradient aus der Loss-Function
        """
        return np.dot(data_input.T, score_gradients)

    def backprop_layer_bias(self, score_gradients):
        """
        Backpropagates den Output durch den Layer bezüglich des Biases.

        Parameter:
          score_gradients - Der Gradient aus der Loss-Function
        """
        return np.sum(score_gradients, axis=0, keepdims=True)


class linearReluClassifier():
    """
    Klasse die einen "simplen" Klassifizierer implementiert,
    dabei wird zusätzlich die Relu-Aktivierungsfunktion nach einem
    Layer angewendet.
    """
    
    def __init__(self, dims, num_classes):
        self.weights = 0.01 * np.random.randn(dims,num_classes)
        self.bias = np.zeros((1,num_classes))

    def forward(self, data_input):
        """
        Berechnet die Forward-Step des Networks.

        Parameter:
          data_input - Input des Datasets
        """
        self.last_input = data_input  # falls benötigt für Backprop
        linear_output = np.dot(data_input, self.weights) + self.bias
        return np.maximum(0, linear_output)
    
    def backprop_relu(self, dscores, probs):
        """
        Backpropagates die Relu Funktion.
        https://yashgarg1232.medium.com/derivative-of-neural-activation-function-64e9e825b67

        Parameter:
          dscores - Gradient der Loss-Function
          probs - Output des Networks
        """
        relu_grad = probs > 0
        return dscores * relu_grad

    def backprop_layer_weights(self, data_input, score_gradients):
        """
        Backpropagates den Output durch den Layer bezüglich der Gewichte.

        Parameter:
          data_input - Input des Datasets
          score_gradients - Der Gradient aus der Loss-Function
        """
        return np.dot(data_input.T, score_gradients)

    def backprop_layer_bias(self, score_gradients):
        """
        Backpropagates den Output durch den Layer bezüglich des Biases.

        Parameter:
          score_gradients - Der Gradient aus der Loss-Function
        """
        return np.sum(score_gradients, axis=0, keepdims=True)

    
class TestCase_linearClassifier(unittest.TestCase):
    
    def setUp(self):
        np.random.seed(42)
        self.c = linearClassifier(2,5)
        
    def test_forward(self):
        i = np.array([[2,3]])
        o = [[0.002910174351749237,
              0.04461109844179805,
              0.035976812636601116,
              0.016376365550111946,
              0.011593733813112222]]
        self.assertEqual(self.c.forward(i).tolist(), o)

    def test_backprop_layer_weights(self):
        d = np.array([[2,3]])
        g = np.array([[5,6]])
        o = [[10, 12], [15, 18]]
        self.assertEqual(self.c.backprop_layer_weights(d, g).tolist(), o)

    def test_backprop_layer_bias(self):
        d = np.array([[2, 3], [3, 5]])
        o = [[5, 8]]
        self.assertEqual(self.c.backprop_layer_bias(d).tolist(), o)

class TestCase_linearReluClassifier(unittest.TestCase):
    
    def setUp(self):
        np.random.seed(42)
        self.c = linearReluClassifier(2,5)

    def test_forward(self):
        i = np.array([[2,3]])
        o = [[0.002910174351749237,
              0.04461109844179805,
              0.035976812636601116,
              0.016376365550111946,
              0.011593733813112222]]
        self.assertEqual(self.c.forward(i).tolist(), o)

    def test_backprop_relu(self):
        d = np.array([[5,3]])
        g = np.array([[-2,6]])
        o = [[0, 3]]
        self.assertEqual(self.c.backprop_relu(d, g).tolist(), o)

    def test_backprop_layer_weights(self):
        d = np.array([[8,3]])
        g = np.array([[6,5]])
        o = [[48, 40], [18, 15]]
        self.assertEqual(self.c.backprop_layer_weights(d, g).tolist(), o)

    def test_backprop_layer_bias(self):
        d = np.array([[2, 3], [5, 3]])
        o = [[7, 6]]
        self.assertEqual(self.c.backprop_layer_bias(d).tolist(), o)

if __name__ == '__main__':
    unittest.main()
