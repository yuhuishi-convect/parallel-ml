# Implementation of a MLP model with 2 hidden layers
# that can do multi-class classification
# using numpy and autograd


import autograd.numpy as np
import autograd.numpy.random as npr
from autograd import grad
from sklearn.datasets import load_iris


# Generate some synthetic data for classification
np.random.seed(42)

data = load_iris()
X = data["data"]
y = data["target"]

num_classes = 3
num_features = X.shape[1]

# Convert labels to one-hot encoding
y_one_hot = np.eye(num_classes)[y].astype(float)

# Neural network architecture
input_size = num_features
hidden_size = 10
output_size = num_classes


# params
params = {
    "W1": np.random.randn(input_size, hidden_size),
    "b1": np.random.randn(hidden_size),
    "W2": np.random.randn(hidden_size, output_size),
    "b2": np.random.randn(output_size),
}

# Activation functions
def relu(x):
    return np.maximum(0, x)


def softmax(x):
    exps = np.exp(x - np.max(x))
    return exps / np.sum(exps)


# Forward pass
def forward_pass(params, X):
    W1 = params["W1"]
    b1 = params["b1"]
    W2 = params["W2"]
    b2 = params["b2"]

    hidden = relu(np.dot(X, W1) + b1)
    scores = np.dot(hidden, W2) + b2
    probs = softmax(scores)

    return probs


# Loss function
def loss(params, X, y):
    probs = forward_pass(params, X)
    return -np.sum(y * np.log(probs))


# Accuracy
def accuracy(params, X, y):
    target_class = np.argmax(y, axis=1)
    predicted_class = np.argmax(forward_pass(params, X), axis=1)
    return (predicted_class == target_class).mean()


# Training
num_epochs = 1000
learning_rate = 1e-3

for i in range(num_epochs):
    params_grad = grad(loss)(params, X, y_one_hot)
    params["W1"] -= learning_rate * params_grad["W1"]
    params["b1"] -= learning_rate * params_grad["b1"]
    params["W2"] -= learning_rate * params_grad["W2"]
    params["b2"] -= learning_rate * params_grad["b2"]

    if i % 100 == 0:
        print("Loss after iteration {}: {}".format(i, loss(params, X, y_one_hot)))
        print(
            "Training accuracy after iteration {}: {}".format(
                i, accuracy(params, X, y_one_hot)
            )
        )

print(
    "Training accuracy after iteration {}: {}".format(
        num_epochs, accuracy(params, X, y_one_hot)
    )
)
