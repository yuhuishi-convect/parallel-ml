# this is a numpy implementation of MLP model

import numpy as np
from sklearn.datasets import make_classification


# MLP implementation using Python and NumPy
class MLP:
    def __init__(self, input_size, hidden_size, output_size):
        self.W1 = np.random.randn(input_size, hidden_size)  # First layer weights
        self.b1 = np.zeros(hidden_size)  # First layer biases
        self.W2 = np.random.randn(hidden_size, output_size)  # Second layer weights
        self.b2 = np.zeros(output_size)  # Second layer biases

    def forward(self, X):
        self.z1 = np.dot(X, self.W1) + self.b1  # Hidden layer linear transformation
        self.a1 = self.sigmoid(self.z1)  # Hidden layer activation
        self.z2 = (
            np.dot(self.a1, self.W2) + self.b2
        )  # Output layer linear transformation
        self.a2 = self.softmax(self.z2)  # Output layer activation
        return self.a2

    def backward(self, X, y, learning_rate):
        m = X.shape[0]

        # Compute gradients
        dZ2 = self.a2 - y
        dW2 = (1 / m) * np.dot(self.a1.T, dZ2)
        db2 = (1 / m) * np.sum(dZ2, axis=0)
        dZ1 = np.dot(dZ2, self.W2.T) * self.sigmoid_derivative(self.z1)
        dW1 = (1 / m) * np.dot(X.T, dZ1)
        db1 = (1 / m) * np.sum(dZ1, axis=0)

        # Update parameters
        self.W2 -= learning_rate * dW2
        self.b2 -= learning_rate * db2
        self.W1 -= learning_rate * dW1
        self.b1 -= learning_rate * db1

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def sigmoid_derivative(self, x):
        return self.sigmoid(x) * (1 - self.sigmoid(x))

    def softmax(self, x):
        exps = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exps / np.sum(exps, axis=1, keepdims=True)

    def predict(self, X):
        return np.argmax(self.forward(X), axis=1)
