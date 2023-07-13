import numpy as np
from sklearn.datasets import make_classification
import pytest
from mlp import MLP


@pytest.fixture
def classification_dataset():
    # fix random seed for reproducibility
    X, y = make_classification(
        n_samples=500, n_features=10, n_classes=2, random_state=42
    )
    # convert y to one-hot encoding
    y = np.eye(2)[y]
    return X, y


def test_train_mlp(classification_dataset):
    X, y = classification_dataset

    split_ratio = 0.8
    split_index = int(split_ratio * X.shape[0])
    X_train, y_train = X[:split_index], y[:split_index]
    X_test, y_test = X[split_index:], y[split_index:]

    # Instantiate the MLP model
    input_size = X.shape[1]
    hidden_size = 64
    output_size = 2
    mlp = MLP(input_size, hidden_size, output_size)

    # Training loop
    epochs = 1000
    learning_rate = 0.1
    for epoch in range(epochs):
        # Forward pass
        y_pred = mlp.forward(X_train)

        # Compute cross-entropy loss
        loss = (-1 / X_train.shape[0]) * np.sum(y_train * np.log(y_pred + 1e-8))

        # Backward pass
        mlp.backward(X_train, y_train, learning_rate)

        # Print loss every 100 epochs
        if epoch % 100 == 0:
            print(f"Epoch {epoch}/{epochs}, Loss: {loss:.4f}")

    # Evaluate the trained model
    y_train_pred = mlp.predict(X_train)
    train_accuracy = np.mean(y_train_pred == np.argmax(y_train, axis=1)) * 100
    print(f"Train Accuracy: {train_accuracy:.2f}%")
    # we should expect at least 90% accuracy on the training set
    assert train_accuracy >= 90

    y_test_pred = mlp.predict(X_test)
    test_accuracy = np.mean(y_test_pred == np.argmax(y_test, axis=1)) * 100
    print(f"Test Accuracy: {test_accuracy:.2f}%")
    # we should expect at least 90% accuracy on the test set
    assert test_accuracy >= 90
