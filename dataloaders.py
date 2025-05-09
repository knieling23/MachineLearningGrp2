import numpy as np
import matplotlib.pyplot as plt
import os

class NonLinearDataset:
    def __init__(self):
        # 2D, 3 Klassen, 300 Samples, nichtlinear verteilt (z.B. Kreise)
        from sklearn.datasets import make_circles
        X, y = make_circles(n_samples=300, noise=0.1, factor=0.5, random_state=42)
        # Für 3 Klassen Dummy: eine Klasse zufällig hinzufügen
        y = np.where(y == 1, 2, y)  # Jetzt: 0 und 2
        # Füge einige zufällig als Klasse 1 hinzu
        idx = np.random.choice(np.where(y == 0)[0], size=100, replace=False)
        y[idx] = 1
        self.data_input = X
        self.data_output = y
        self.fig_save = "C:/tmp/fig_output.png"
        self.Dim = self.data_input.shape[1]
        self.K = 3

    def eval_network(self, network):
        predictions = network.forward(self.data_input)
        predicted_classes = np.argmax(predictions, axis=1)
        if self.data_output.ndim == 2:
            true_classes = np.argmax(self.data_output, axis=1)
        else:
            true_classes = self.data_output
        accuracy = np.mean(predicted_classes == true_classes)
        print(f"Accuracy: {accuracy * 100:.2f}%")

    def display_classification_result(self, network):
        predictions = network.forward(self.data_input)
        predicted_classes = np.argmax(predictions, axis=1)
        plt.figure(figsize=(8, 6))
        scatter = plt.scatter(self.data_input[:, 0], self.data_input[:, 1], c=predicted_classes, cmap="viridis", s=40)
        plt.title("Klassifikationsergebnis")
        plt.xlabel("Feature 1")
        plt.ylabel("Feature 2")
        plt.colorbar(scatter, label="Vorhergesagte Klasse")
        fig = plt.gcf()
        save_dir = os.path.dirname(self.fig_save)
        if save_dir and not os.path.exists(save_dir):
            os.makedirs(save_dir)
        fig.savefig(self.fig_save)
        plt.close(fig)

class LinearDataset:
    def __init__(self):
        # 2D, 3 Klassen, 300 Samples, linear trennbar
        from sklearn.datasets import make_classification
        X, y = make_classification(n_samples=300, n_features=2, n_informative=2, n_redundant=0,
                                   n_classes=3, n_clusters_per_class=1, random_state=42)
        self.data_input = X
        self.data_output = y
        self.fig_save = "C:/tmp/fig_output.png"
        self.Dim = self.data_input.shape[1]
        self.K = 3

    def eval_network(self, network):
        predictions = network.forward(self.data_input)
        predicted_classes = np.argmax(predictions, axis=1)
        if self.data_output.ndim == 2:
            true_classes = np.argmax(self.data_output, axis=1)
        else:
            true_classes = self.data_output
        accuracy = np.mean(predicted_classes == true_classes)
        print(f"Accuracy: {accuracy * 100:.2f}%")

    def display_classification_result(self, network):
        predictions = network.forward(self.data_input)
        predicted_classes = np.argmax(predictions, axis=1)
        plt.figure(figsize=(8, 6))
        scatter = plt.scatter(self.data_input[:, 0], self.data_input[:, 1], c=predicted_classes, cmap="viridis", s=40)
        plt.title("Klassifikationsergebnis")
        plt.xlabel("Feature 1")
        plt.ylabel("Feature 2")
        plt.colorbar(scatter, label="Vorhergesagte Klasse")
        fig = plt.gcf()
        save_dir = os.path.dirname(self.fig_save)
        if save_dir and not os.path.exists(save_dir):
            os.makedirs(save_dir)
        fig.savefig(self.fig_save)
        plt.close(fig)

class OneHotNonLinearDataset:
    def __init__(self):
        # Wie NonLinearDataset, aber Labels als One-Hot
        from sklearn.datasets import make_circles
        X, y = make_circles(n_samples=300, noise=0.1, factor=0.5, random_state=42)
        y = np.where(y == 1, 2, y)
        idx = np.random.choice(np.where(y == 0)[0], size=100, replace=False)
        y[idx] = 1
        K = 3
        y_onehot = np.zeros((y.size, K))
        y_onehot[np.arange(y.size), y] = 1
        self.data_input = X
        self.data_output = y_onehot
        self.fig_save = "C:/tmp/fig_output.png"
        self.Dim = self.data_input.shape[1]
        self.K = K

    def eval_network(self, network):
        predictions = network.forward(self.data_input)
        predicted_classes = np.argmax(predictions, axis=1)
        if self.data_output.ndim == 2:
            true_classes = np.argmax(self.data_output, axis=1)
        else:
            true_classes = self.data_output
        accuracy = np.mean(predicted_classes == true_classes)
        print(f"Accuracy: {accuracy * 100:.2f}%")

    def display_classification_result(self, network):
        predictions = network.forward(self.data_input)
        predicted_classes = np.argmax(predictions, axis=1)
        plt.figure(figsize=(8, 6))
        scatter = plt.scatter(self.data_input[:, 0], self.data_input[:, 1], c=predicted_classes, cmap="viridis", s=40)
        plt.title("Klassifikationsergebnis")
        plt.xlabel("Feature 1")
        plt.ylabel("Feature 2")
        plt.colorbar(scatter, label="Vorhergesagte Klasse")
        fig = plt.gcf()
        save_dir = os.path.dirname(self.fig_save)
        if save_dir and not os.path.exists(save_dir):
            os.makedirs(save_dir)
        fig.savefig(self.fig_save)
        plt.close(fig)

class OneHotLinearDataset:
    def __init__(self):
        # Wie LinearDataset, aber Labels als One-Hot
        from sklearn.datasets import make_classification
        X, y = make_classification(n_samples=300, n_features=2, n_informative=2, n_redundant=0,
                                   n_classes=3, n_clusters_per_class=1, random_state=42)
        K = 3
        y_onehot = np.zeros((y.size, K))
        y_onehot[np.arange(y.size), y] = 1
        self.data_input = X
        self.data_output = y_onehot
        self.fig_save = "C:/tmp/fig_output.png"
        self.Dim = self.data_input.shape[1]
        self.K = K

    def eval_network(self, network):
        predictions = network.forward(self.data_input)
        predicted_classes = np.argmax(predictions, axis=1)
        if self.data_output.ndim == 2:
            true_classes = np.argmax(self.data_output, axis=1)
        else:
            true_classes = self.data_output
        accuracy = np.mean(predicted_classes == true_classes)
        print(f"Accuracy: {accuracy * 100:.2f}%")

    def display_classification_result(self, network):
        predictions = network.forward(self.data_input)
        predicted_classes = np.argmax(predictions, axis=1)
        plt.figure(figsize=(8, 6))
        scatter = plt.scatter(self.data_input[:, 0], self.data_input[:, 1], c=predicted_classes, cmap="viridis", s=40)
        plt.title("Klassifikationsergebnis")
        plt.xlabel("Feature 1")
        plt.ylabel("Feature 2")
        plt.colorbar(scatter, label="Vorhergesagte Klasse")
        fig = plt.gcf()
        save_dir = os.path.dirname(self.fig_save)
        if save_dir and not os.path.exists(save_dir):
            os.makedirs(save_dir)
        fig.savefig(self.fig_save)
        plt.close(fig)
