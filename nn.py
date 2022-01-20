import numpy as np


class NeuralNetwork:

    def __init__(self, layer_sizes, learning_rate=1):
        """
        Neural Network initialization.
        Given layer_sizes as an input, you have to design a Fully Connected Neural Network architecture here.
        :param layer_sizes: A list containing neuron numbers in each layers. For example [3, 10, 2] means that there are
        3 neurons in the input layer, 10 neurons in the hidden layer, and 2 neurons in the output layer.
        """
        self.weights = []
        self.biases = []
        self.learning_rate = learning_rate

        layers_size = []
        for i, size in enumerate(layer_sizes):
            if i == 1:
                continue
            prev = layer_sizes[i-1]
            layers_size.append((prev, size))

        for size in layers_size:
            self.weights.append(np.random.normal(size=size))
            self.biases.append(np.zeros((1, size[1])).astype(np.longdouble))

        self.layers_number = len(self.weights)

    def activation(self, x):
        """
        The activation function of our neural network, e.g., Sigmoid, ReLU.
        :param x: Vector of a layer in our network.
        :return: Vector after applying activation function.
        """
        return 1/(1 + np.exp(-x, dtype=np.longdouble))

    def forward(self, x):
        """
        Receives input vector as a parameter and calculates the output vector based on weights and biases.
        :param x: Input vector which is a numpy array.
        :return: Output vector
        """
        W = self.weights
        b = self.biases

        result = []
        o = self.activation(W[0].T.dot(x) + b[0].T)
        result.append(o)
        for i in range(1, self.layers_number):
            o = self.activation(W[i].T.dot(o) + b[i].T)
            result.append(o)
        return result
