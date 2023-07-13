from matrix_op import (
    matrix_addition,
    matrix_multiplication_blockwise,
    matrix_multiplication_rowwise,
)
import pytest
import numpy as np

# fix the random seed for reproducibility
np.random.seed(0)


@pytest.fixture
def matrix_a():
    return np.random.randn(100, 100)


@pytest.fixture
def matrix_b():
    return np.random.randn(100, 100)


def test_matrix_addition(matrix_a, matrix_b):
    matrix_sum = matrix_addition(matrix_a, matrix_b)
    expected_sum = matrix_a + matrix_b
    print(matrix_sum - expected_sum)
    assert np.allclose(matrix_sum, expected_sum)


def test_matrix_multiplication_rowwise(matrix_a, matrix_b):
    matrix_product = matrix_multiplication_rowwise(matrix_a, matrix_b)

    expected_product = matrix_a @ matrix_b

    assert np.allclose(matrix_product, expected_product)


def test_matrix_multiplication_blockwise(matrix_a, matrix_b):
    matrix_product = matrix_multiplication_blockwise(matrix_a, matrix_b)
    expected_product = matrix_a @ matrix_b
    assert np.allclose(matrix_product, expected_product)
