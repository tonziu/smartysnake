import numpy as np


class Network:
    def __init__(self):
        self.architecture = []
        self.weights = []
        self.biases = []

    def init_layers(self):
        for i in range(self.num_layers):
            w = np.random.normal(0, 1/(self.architecture[i]), size=(self.architecture[i], self.architecture[i+1]))
            b = np.zeros(shape=(1, self.architecture[i+1]))
            self.weights.append(w)
            self.biases.append(b)

    def feed(self, x):
        x = np.array(x)
        for w,b in zip(self.weights[:-1], self.biases[:-1]):
            y = self.activate(x@w + b, hidden=True)
            x = y
        y = self.activate(x@self.weights[-1] + self.biases[-1], hidden=False)
        return y

    @property
    def num_layers(self):
        return len(self.architecture)-1

    @staticmethod
    def activate(x, hidden):
        if hidden:
            return np.tanh(x)
        return np.maximum(0, x)

    @staticmethod
    def default():
        new_network = Network()
        new_network.architecture = [16, 32, 4]
        return new_network

    @staticmethod
    def copy_from(other):
        new_network = Network()
        new_network.architecture = other.architecture
        return new_network


if __name__ == "__main__":
    network = Network.default()
    network.init_layers()
    x = np.ones(shape=(1, 32))
    y = network.feed(x)
