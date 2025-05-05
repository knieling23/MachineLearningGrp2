import numpy as np
import matplotlib.pyplot as plt

"""
Dieses Modul berechnet das Dataset.
Die folgenden Klassenvariablen sind in allen Klassen vorhanden:

- data_input: Eine Liste mit allen Elementen des Inputs. Jedes Element
              ist eine Liste.

- data_output: Eine Liste mit der jeweiligen Klasse des data_inputs.
               Je nach Klasse als Index oder One-Hot Encoded.

- K: Die Anzahl der Klassen.

- Dim: Die Dimension eines Inputs.
"""



class NonLinearDataset():
    """ Klasse zum erstellen eines Datensets, welches nicht mit einem
        Linearen Klassifizierer klassifiziert werden kann.
    """
    
    def __init__(self, fig_save="/tmp/fig_output.png"):

        N = 100 # samples per class to create
        
        self.Dim = 2 
        self.K = 3 
        self.data_input = np.zeros((N*self.K,self.Dim)) 
        self.data_output = np.zeros(N*self.K, dtype='uint8')

        self.fig_save = fig_save
        
        for c in range(self.K):
            idx = range(N*c,N*(c+1))
            radius = np.linspace(0.0,1,N)
            theta = np.linspace(c*4,(c+1)*4,N) + np.random.randn(N)*0.2
            self.data_input[idx] = np.c_[radius*np.sin(theta), radius*np.cos(theta)]
            self.data_output[idx] = c

    def display_dataset(self):
        """
        Erstellt ein grafische Repräsentation des Datensatzes.
        """
        fig = plt.figure()
        plt.scatter(self.data_input[:, 0], self.data_input[:, 1], c=self.data_output, s=40)
        plt.xlim([-1,1])
        plt.ylim([-1,1])
        fig.savefig(self.fig_save)

    def display_classification_result(self, network):
        """
        Erstellt ein grafische Repräsentation des Datensatzes und der Ergebnisse des Networks.

        Parameter:
           network - Das zu testende Network
        """
        h = 0.02
        x_min, x_max = self.data_input[:, 0].min() - 1, self.data_input[:, 0].max() + 1
        y_min, y_max = self.data_input[:, 1].min() - 1, self.data_input[:, 1].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                             np.arange(y_min, y_max, h))
        Z = network.forward(np.c_[xx.ravel(), yy.ravel()])
        Z = np.argmax(Z, axis=1)
        Z = Z.reshape(xx.shape)
        fig = plt.figure()
        plt.contourf(xx, yy, Z, alpha=0.8)
        plt.scatter(self.data_input[:, 0], self.data_input[:, 1], c=self.data_output, s=40)
        plt.xlim(xx.min(), xx.max())
        plt.ylim(yy.min(), yy.max())
        fig.savefig(self.fig_save)

    def eval_network(self, network):
        """
        Berechnet die Accuracy eines Networks angewendet auf diesen Datenset.
        Das Ergebnis wird auf std_out über die python `print` Funktion ausgegeben.

        Paramter:
          network - Das zu testende Network
        """
        pass
        
        
class LinearDataset():
    """ Klasse zum erstellen eines Datensets, welches mit einem
        Linearen Klassifizierer klassifiziert werden kann.
    """
    
    def __init__(self, fig_save="/tmp/fig_output.png"):

        N = 100 # samples per class to create
        
        self.Dim = 2 
        self.K = 3 
        self.data_input = np.zeros((N*self.K,self.Dim)) 
        self.data_output = np.zeros(N*self.K, dtype='uint8')

        self.fig_save = fig_save

        for c in range(self.K):
            idx = range(N*c,N*(c+1))
            radius = np.linspace(0.0,1,N)
            theta = np.linspace(c*4,(c)*4,N) + np.random.randn(N)*0.2
            self.data_input[idx] = np.c_[radius*np.sin(theta), radius*np.cos(theta)]
            self.data_output[idx] = c

    def display_dataset(self):
        """
        Erstellt ein grafische Repräsentation des Datensatzes.
        """
        fig = plt.figure()
        plt.scatter(self.data_input[:, 0], self.data_input[:, 1], c=self.data_output, s=40)
        plt.xlim([-1,1])
        plt.ylim([-1,1])
        fig.savefig(self.fig_save)

    def display_classification_result(self, network):
        """
        Erstellt ein grafische Repräsentation des Datensatzes und der Ergebnisse des Networks.

        Parameter:
           network - Das zu testende Network
        """
        h = 0.02
        x_min, x_max = self.data_input[:, 0].min() - 1, self.data_input[:, 0].max() + 1
        y_min, y_max = self.data_input[:, 1].min() - 1, self.data_input[:, 1].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                             np.arange(y_min, y_max, h))
        Z = network.forward(np.c_[xx.ravel(), yy.ravel()])
        Z = np.argmax(Z, axis=1)
        Z = Z.reshape(xx.shape)
        fig = plt.figure()
        plt.contourf(xx, yy, Z, alpha=0.8)
        plt.scatter(self.data_input[:, 0], self.data_input[:, 1], c=self.data_output, s=40)
        plt.xlim(xx.min(), xx.max())
        plt.ylim(yy.min(), yy.max())
        fig.savefig(self.fig_save)

    def eval_network(self, network):
        """
        Berechnet die Accuracy eines Networks angewendet auf diesen Datenset.
        Das Ergebnis wird auf std_out über die python `print` Funktion ausgegeben.

        Paramter:
          network - Das zu testende Network
        """
        pass
        

class OneHotNonLinearDataset():
    """ Klasse zum erstellen eines Datensets, welches nicht mit einem
        Linearen Klassifizierer klassifiziert werden kann.
    """
    
    def __init__(self, fig_save="/tmp/fig_output.png"):

        N = 100 # samples per class to create
        
        self.Dim = 2 
        self.K = 3 
        self.data_input = np.zeros((N*self.K,self.Dim)) 
        self.data_output = np.zeros((N*self.K, self.K))

        self.fig_save = fig_save

        for c in range(self.K):
            idx = range(N*c,N*(c+1))
            radius = np.linspace(0.0,1,N)
            theta = np.linspace(c*4,(c+1)*4,N) + np.random.randn(N)*0.2
            self.data_input[idx] = np.c_[radius*np.sin(theta), radius*np.cos(theta)]
            a = np.zeros(self.K)
            a[c] = 1
            self.data_output[idx] = a

    def display_dataset(self):
        """
        Erstellt ein grafische Repräsentation des Datensatzes.
        """
        fig = plt.figure()
        plt.scatter(self.data_input[:, 0], self.data_input[:, 1], c=np.argmax(self.data_output, axis=1), s=40)
        plt.xlim([-1,1])
        plt.ylim([-1,1])
        fig.savefig(self.fig_save)

    def display_classification_result(self, network):
        """
        Erstellt ein grafische Repräsentation des Datensatzes und der Ergebnisse des Networks.

        Parameter:
           network - Das zu testende Network
        """
        h = 0.02
        x_min, x_max = self.data_input[:, 0].min() - 1, self.data_input[:, 0].max() + 1
        y_min, y_max = self.data_input[:, 1].min() - 1, self.data_input[:, 1].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                             np.arange(y_min, y_max, h))
        Z = network.forward(np.c_[xx.ravel(), yy.ravel()])
        Z = np.argmax(Z, axis=1)
        Z = Z.reshape(xx.shape)
        fig = plt.figure()
        plt.contourf(xx, yy, Z, alpha=0.8)
        plt.scatter(self.data_input[:, 0], self.data_input[:, 1], c=np.argmax(self.data_output, axis=1), s=40)
        plt.xlim(xx.min(), xx.max())
        plt.ylim(yy.min(), yy.max())
        fig.savefig(self.fig_save)

    def eval_network(self, network):
        """
        Berechnet die Accuracy eines Networks angewendet auf diesen Datenset.
        Das Ergebnis wird auf std_out über die python `print` Funktion ausgegeben.

        Paramter:
          network - Das zu testende Network
        """
        pass


class OneHotLinearDataset():
    """ Klasse zum erstellen eines Datensets, welches mit einem
        Linearen Klassifizierer klassifiziert werden kann.
    """
    
    def __init__(self, fig_save="/tmp/fig_output.png"):

        N = 100 # samples per class to create
        
        self.Dim = 2 
        self.K = 3 
        self.data_input = np.zeros((N*self.K,self.Dim)) 
        self.data_output = np.zeros((N*self.K, self.K))

        self.fig_save = fig_save

        for c in range(self.K):
            idx = range(N*c,N*(c+1))
            radius = np.linspace(0.0,1,N)
            theta = np.linspace(c*4,(c)*4,N) + np.random.randn(N)*0.2
            self.data_input[idx] = np.c_[radius*np.sin(theta), radius*np.cos(theta)]
            a = np.zeros(self.K)
            a[c] = 1
            self.data_output[idx] = a

    def display_dataset(self):
        """
        Erstellt ein grafische Repräsentation des Datensatzes.
        """
        fig = plt.figure()
        plt.scatter(self.data_input[:, 0], self.data_input[:, 1], c=np.argmax(self.data_output, axis=1), s=40)
        plt.xlim([-1,1])
        plt.ylim([-1,1])
        fig.savefig(self.fig_save)

    def display_classification_result(self, network):
        """
        Erstellt ein grafische Repräsentation des Datensatzes und der Ergebnisse des Networks.

        Parameter:
           network - Das zu testende Network
        """
        h = 0.02
        x_min, x_max = self.data_input[:, 0].min() - 1, self.data_input[:, 0].max() + 1
        y_min, y_max = self.data_input[:, 1].min() - 1, self.data_input[:, 1].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                             np.arange(y_min, y_max, h))
        Z = network.forward(np.c_[xx.ravel(), yy.ravel()])
        Z = np.argmax(Z, axis=1)
        Z = Z.reshape(xx.shape)
        fig = plt.figure()
        plt.contourf(xx, yy, Z, alpha=0.8)
        plt.scatter(self.data_input[:, 0], self.data_input[:, 1], c=np.argmax(self.data_output, axis=1), s=40)
        plt.xlim(xx.min(), xx.max())
        plt.ylim(yy.min(), yy.max())
        fig.savefig(self.fig_save)

    def eval_network(self, network):
        """
        Berechnet die Accuracy eines Networks angewendet auf diesen Datenset.
        Das Ergebnis wird auf std_out über die python `print` Funktion ausgegeben.

        Paramter:
          network - Das zu testende Network
        """
        pass
